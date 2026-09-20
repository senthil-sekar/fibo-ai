//
//  VectorDBService.swift
//  Fibo
//
//  Thin wrapper around LocalVectorStore: handles embedding text before storage
//  and shaping search results as RetrievedChunk. Chunking is owned by the
//  caller (RAGService) — this stays a one-document-in, one-document-out API.
//

import Foundation

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
class VectorDBService: ObservableObject {
    static let shared = VectorDBService()

    @Published var lastError: String?

    private static let sourceDateKey = "sourceDate"
    private static let subtitleKey = "subtitle"

    private init() {}

    // MARK: - Upsert

    /// Embed and store a single chunk. `id` should be `"\(sourceId)#\(index)"`
    /// for multi-chunk sources so `deleteSource` can remove them all together.
    func upsert(
        id: String,
        content: String,
        type: String,
        title: String,
        subtitle: String = "",
        sourceDate: Date? = nil
    ) async throws {
        let vector = try EmbeddingService.embedLocally(content)
        var metadata: [String: String] = [:]
        if !subtitle.isEmpty { metadata[Self.subtitleKey] = subtitle }
        if let sourceDate {
            metadata[Self.sourceDateKey] = ISO8601DateFormatter().string(from: sourceDate)
        }
        LocalVectorStore.shared.upsert(LocalVectorStore.VectorDocument(
            id: id, type: type, title: title,
            content: content, vector: vector, indexedAt: Date(),
            metadata: metadata
        ))
        lastError = nil
    }

    // MARK: - Delete

    func delete(id: String) {
        LocalVectorStore.shared.delete(id: id)
    }

    /// Remove every chunk belonging to a source (all ids sharing its `"\(sourceId)#"` prefix).
    func deleteSource(_ sourceId: String) {
        LocalVectorStore.shared.deleteAll(withPrefix: "\(sourceId)#")
    }

    func deleteAll() {
        LocalVectorStore.shared.deleteAll()
    }

    // MARK: - Search

    func search(
        query: String,
        type: String? = nil,
        topK: Int = Configuration.RAG.topK
    ) async throws -> [RetrievedChunk] {
        let queryVector = try EmbeddingService.embedLocally(query)
        let matches = LocalVectorStore.shared.search(
            queryVector: queryVector, topK: topK, typeFilter: type
        )
        lastError = nil
        return matches.map { match in
            let sourceDate = match.metadata[Self.sourceDateKey]
                .flatMap { ISO8601DateFormatter().date(from: $0) }
            let sourceId = match.id.split(separator: "#", maxSplits: 1).first.map(String.init) ?? match.id
            return RetrievedChunk(
                id: match.id,
                sourceId: sourceId,
                type: match.type,
                title: match.title,
                text: match.content,
                subtitle: match.metadata[Self.subtitleKey] ?? "",
                sourceDate: sourceDate,
                score: match.score
            )
        }
    }

    /// Count of indexed chunks (for stats / debugging).
    func chunkCount() -> Int {
        LocalVectorStore.shared.documentCount
    }
}
