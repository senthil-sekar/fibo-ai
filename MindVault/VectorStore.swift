//
//  VectorStore.swift
//  MindVault
//
//  On-device vector store: embeds content with the EmbeddingEngine and persists
//  it as IndexedChunk rows in SwiftData. Retrieval is brute-force cosine
//  similarity — more than fast enough for a personal corpus, and dependency-free.
//

import Foundation
import SwiftData

/// A chunk returned from a similarity search, with its relevance score.
struct RetrievedChunk: Identifiable {
    let id: String
    let sourceId: String
    let type: String
    let title: String
    let text: String
    let subtitle: String
    let sourceDate: Date?
    let score: Float
}

@MainActor
final class VectorStore {
    private let modelContext: ModelContext
    private let embedder: EmbeddingEngine

    init(modelContext: ModelContext, embedder: EmbeddingEngine) {
        self.modelContext = modelContext
        self.embedder = embedder
    }

    // MARK: - Indexing

    /// Embed and store an item. Existing chunks for the same `sourceId` are
    /// replaced, so this is safe to call on every edit (idempotent upsert).
    /// Returns the number of chunks stored.
    @discardableResult
    func index(
        sourceId: String,
        type: String,
        title: String,
        fullText: String,
        subtitle: String = "",
        sourceDate: Date? = nil
    ) async throws -> Int {
        deleteSource(sourceId)

        let chunks = Self.chunk(fullText)
        guard !chunks.isEmpty else { return 0 }

        let vectors = try await embedder.embed(chunks, isQuery: false)

        for (i, (chunkText, vector)) in zip(chunks, vectors).enumerated() {
            let record = IndexedChunk(
                id: "\(sourceId)#\(i)",
                sourceId: sourceId,
                type: type,
                title: title,
                text: chunkText,
                subtitle: subtitle,
                sourceDate: sourceDate,
                vector: Self.normalize(vector)
            )
            modelContext.insert(record)
        }
        try modelContext.save()
        return chunks.count
    }

    // MARK: - Deletion

    /// Remove all chunks belonging to a source item.
    func deleteSource(_ sourceId: String) {
        let descriptor = FetchDescriptor<IndexedChunk>(
            predicate: #Predicate { $0.sourceId == sourceId }
        )
        if let existing = try? modelContext.fetch(descriptor) {
            for chunk in existing { modelContext.delete(chunk) }
        }
    }

    /// Remove everything (e.g. after switching embedding models — vectors from a
    /// different model aren't comparable).
    func deleteAll() throws {
        try modelContext.delete(model: IndexedChunk.self)
        try modelContext.save()
    }

    // MARK: - Search

    /// Return the most relevant chunks for a query, optionally restricted to a
    /// single source `type` ("email", "journal", …).
    func search(
        _ query: String,
        type: String? = nil,
        topK: Int = OnDeviceConfig.topK,
        minScore: Float = OnDeviceConfig.minRelevanceScore
    ) async throws -> [RetrievedChunk] {
        let queryVector = Self.normalize(try await embedder.embed(query, isQuery: true))
        guard !queryVector.isEmpty else { return [] }
        let dim = queryVector.count

        var descriptor = FetchDescriptor<IndexedChunk>()
        if let type {
            descriptor.predicate = #Predicate { $0.type == type }
        }
        let candidates = try modelContext.fetch(descriptor)

        let scored: [RetrievedChunk] = candidates.compactMap { chunk in
            guard chunk.dimension == dim else { return nil }   // skip stale-model rows
            let score = Self.dot(queryVector, chunk.vector)
            guard score >= minScore else { return nil }
            return RetrievedChunk(
                id: chunk.id,
                sourceId: chunk.sourceId,
                type: chunk.type,
                title: chunk.title,
                text: chunk.text,
                subtitle: chunk.subtitle,
                sourceDate: chunk.sourceDate,
                score: score
            )
        }

        return Array(scored.sorted { $0.score > $1.score }.prefix(topK))
    }

    /// Count of indexed chunks (for stats / debugging).
    func chunkCount() -> Int {
        (try? modelContext.fetchCount(FetchDescriptor<IndexedChunk>())) ?? 0
    }

    // MARK: - Math

    /// Dot product of two equal-length vectors. With normalized inputs this is
    /// cosine similarity.
    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var sum: Float = 0
        for i in 0..<a.count { sum += a[i] * b[i] }
        return sum
    }

    /// Scale a vector to unit length. Pooled embeddings are usually already
    /// normalized; this guards against models that aren't.
    static func normalize(_ v: [Float]) -> [Float] {
        var norm: Float = 0
        for x in v { norm += x * x }
        norm = norm.squareRoot()
        guard norm > 1e-6 else { return v }
        return v.map { $0 / norm }
    }

    // MARK: - Chunking

    /// Split text into overlapping, sentence-aware chunks. Short text returns a
    /// single chunk.
    static func chunk(
        _ text: String,
        size: Int = OnDeviceConfig.chunkSize,
        overlap: Int = OnDeviceConfig.chunkOverlap
    ) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard trimmed.count > size else { return [trimmed] }

        let chars = Array(trimmed)
        var chunks: [String] = []
        var start = 0

        while start < chars.count {
            var end = min(start + size, chars.count)

            // Try to end on a sentence/whitespace boundary for cleaner chunks.
            if end < chars.count {
                let windowStart = max(start + size - overlap, start)
                if let boundary = (windowStart..<end).reversed().first(where: {
                    let c = chars[$0]
                    return c == "." || c == "\n" || c == "!" || c == "?"
                }) {
                    end = boundary + 1
                }
            }

            let piece = String(chars[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { chunks.append(piece) }

            if end >= chars.count { break }
            start = max(end - overlap, start + 1)
        }
        return chunks
    }
}
