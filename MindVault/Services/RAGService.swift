//
//  RAGService.swift
//  MindVault
//
//  On-device Retrieval-Augmented Generation. Orchestrates the local
//  EmbeddingEngine + VectorStore (retrieval) and LLMEngine (generation).
//  No backend, no network at inference time.
//

import Foundation
import SwiftData

@MainActor
final class RAGService: ObservableObject {
    static let shared = RAGService()

    @Published var isProcessing = false
    @Published var processingStatus: String = ""
    @Published var lastError: String?

    private let models = ModelManager.shared
    private var vectorStore: VectorStore?

    private init() {}

    /// Streaming events emitted while answering a query.
    enum RAGEvent {
        case sources([ChatContext])
        case token(String)
    }

    enum RAGError: LocalizedError {
        case notConfigured
        case modelsNotReady
        var errorDescription: String? {
            switch self {
            case .notConfigured: return "The on-device store isn't ready yet."
            case .modelsNotReady: return "The AI models are still loading. Please wait a moment."
            }
        }
    }

    // MARK: - Wiring

    /// Must be called once at app start with the SwiftData context, so the
    /// VectorStore can read/write IndexedChunk rows.
    func configure(modelContext: ModelContext) {
        if vectorStore == nil {
            vectorStore = VectorStore(modelContext: modelContext, embedder: models.embedder)
        }
    }

    // MARK: - Indexing

    func embedJournalEntry(_ entry: JournalEntry) async {
        guard let vectorStore else { return }
        isProcessing = true
        processingStatus = "Indexing entry…"
        do {
            try await vectorStore.index(
                sourceId: entry.id.uuidString,
                type: "journal",
                title: entry.title.isEmpty ? "Journal Entry" : entry.title,
                fullText: entry.fullText,
                subtitle: entry.category,
                sourceDate: entry.createdAt
            )
            entry.isEmbedded = true
            entry.embeddingId = entry.id.uuidString
            processingStatus = "Indexed"
            lastError = nil
        } catch {
            entry.isEmbedded = false
            lastError = error.localizedDescription
            processingStatus = "Indexing failed"
        }
        isProcessing = false
    }

    func embedProfileItem(_ item: ProfileItem) async {
        guard let vectorStore else { return }
        isProcessing = true
        processingStatus = "Indexing profile item…"
        do {
            try await vectorStore.index(
                sourceId: item.id.uuidString,
                type: "profile",
                title: item.title.isEmpty ? item.type : item.title,
                fullText: item.fullText,
                subtitle: item.subtitle ?? item.type,
                sourceDate: item.startDate
            )
            item.isEmbedded = true
            item.embeddingId = item.id.uuidString
            processingStatus = "Indexed"
            lastError = nil
        } catch {
            item.isEmbedded = false
            lastError = error.localizedDescription
            processingStatus = "Indexing failed"
        }
        isProcessing = false
    }

    func deleteEntry(_ entry: JournalEntry) async {
        vectorStore?.deleteSource(entry.id.uuidString)
    }

    func deleteProfileItem(_ item: ProfileItem) async {
        vectorStore?.deleteSource(item.id.uuidString)
    }

    /// Generic indexing entry point used by Email/Drive ingestion.
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
        guard let vectorStore else { throw RAGError.notConfigured }
        return try await vectorStore.index(
            sourceId: sourceId,
            type: type,
            title: title,
            fullText: fullText,
            subtitle: subtitle,
            sourceDate: sourceDate
        )
    }

    /// Remove all chunks for a source (journal/profile/email/document).
    func removeSource(_ sourceId: String) {
        vectorStore?.deleteSource(sourceId)
    }

    /// Wipe the entire on-device index (used by "Clear AI Data").
    func clearIndex() {
        try? vectorStore?.deleteAll()
    }

    /// Whether indexing/search can run right now.
    var isReady: Bool { vectorStore != nil && models.isReady }

    // MARK: - Retrieval + Generation (streaming)

    /// Stream an answer: first a `.sources` event with the retrieved citations,
    /// then `.token` events as the model generates.
    func streamResponse(
        to query: String,
        history: [(role: String, content: String)] = []
    ) -> AsyncThrowingStream<RAGEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let vectorStore else {
                    continuation.finish(throwing: RAGError.notConfigured); return
                }
                guard models.isReady else {
                    continuation.finish(throwing: RAGError.modelsNotReady); return
                }
                do {
                    let hits = try await vectorStore.search(query)
                    let contexts = hits.map { hit in
                        ChatContext(
                            documentId: hit.sourceId,
                            documentType: hit.type,
                            title: hit.title,
                            snippet: hit.text.count > 300
                                ? String(hit.text.prefix(300)) + "…"
                                : hit.text,
                            relevanceScore: hit.score,
                            date: hit.sourceDate
                        )
                    }
                    continuation.yield(.sources(contexts))

                    guard !hits.isEmpty else {
                        continuation.yield(.token(
                            "I couldn't find anything relevant in your journal, profile, "
                            + "emails, or documents. Try rephrasing, or add more entries first."
                        ))
                        continuation.finish(); return
                    }

                    let contextBlock = Self.buildContextBlock(hits)
                    let userMessage = "CONTEXT:\n\(contextBlock)\n\nQUESTION: \(query)"

                    let stream = await models.llm.generate(
                        system: OnDeviceConfig.systemPrompt,
                        user: userMessage,
                        history: history
                    )
                    for try await delta in stream {
                        continuation.yield(.token(delta))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Non-streaming convenience kept for existing callers: collects the full
    /// streamed answer and its sources.
    func generateResponse(to query: String) async -> (String, [ChatContext]) {
        var text = ""
        var contexts: [ChatContext] = []
        do {
            for try await event in streamResponse(to: query) {
                switch event {
                case .sources(let c): contexts = c
                case .token(let t): text += t
                }
            }
        } catch {
            lastError = error.localizedDescription
            text = "Sorry — \(error.localizedDescription)"
        }
        return (text, contexts)
    }

    // MARK: - Batch (re)index

    /// Index any journal entries / profile items not yet embedded. Called when
    /// models become ready, or from a manual "rebuild index" action.
    @discardableResult
    func syncAllEntries(
        entries: [JournalEntry],
        items: [ProfileItem]
    ) async -> (success: Int, failed: Int) {
        isProcessing = true
        var success = 0, failed = 0
        let total = entries.filter { !$0.isEmbedded }.count
            + items.filter { !$0.isEmbedded }.count
        var index = 0

        for entry in entries where !entry.isEmbedded {
            index += 1
            processingStatus = "Indexing \(index)/\(total)…"
            await embedJournalEntry(entry)
            entry.isEmbedded ? (success += 1) : (failed += 1)
        }
        for item in items where !item.isEmbedded {
            index += 1
            processingStatus = "Indexing \(index)/\(total)…"
            await embedProfileItem(item)
            item.isEmbedded ? (success += 1) : (failed += 1)
        }

        processingStatus = "Index up to date"
        isProcessing = false
        return (success, failed)
    }

    // MARK: - Insights

    func generateInsights(from entries: [JournalEntry]) async -> String? {
        guard !entries.isEmpty, models.isReady else { return nil }
        isProcessing = true
        processingStatus = "Analyzing your journal…"
        defer { isProcessing = false }

        let recent = entries.prefix(20).map { $0.fullText }.joined(separator: "\n\n---\n\n")
        let prompt = """
        Based on these recent journal entries, share 3–5 specific, actionable \
        insights about patterns, growth, or recurring themes.

        Entries:
        \(recent)
        """
        do {
            return try await models.llm.generateComplete(
                system: OnDeviceConfig.systemPrompt,
                user: prompt,
                history: []
            )
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }

    // MARK: - Prompt building

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        return f
    }()

    private static func buildContextBlock(_ hits: [RetrievedChunk]) -> String {
        hits.prefix(OnDeviceConfig.maxContextChunks).enumerated().map { index, hit in
            let label: String
            switch hit.type {
            case "email":
                label = "Email — \"\(hit.title)\""
                    + (hit.subtitle.isEmpty ? "" : " from \(hit.subtitle)")
            case "journal":   label = "Journal — \"\(hit.title)\""
            case "profile":   label = "Profile — \(hit.title)"
            case "document":  label = "Document — \(hit.title)"
            default:          label = hit.title
            }
            let dateStr = hit.sourceDate.map { " [\(dateFormatter.string(from: $0))]" } ?? ""
            return "[\(index + 1)] \(label)\(dateStr)\n\(hit.text)"
        }
        .joined(separator: "\n\n")
    }
}
