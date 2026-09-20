//
//  RAGService.swift
//  Fibo
//
//  On-device Retrieval-Augmented Generation. Orchestrates VectorDBService
//  (retrieval, always on-device in both AI modes) and the active LLMProvider
//  (generation — on-device MLX or OpenAI BYOK, per Settings → AI Mode).
//

import Foundation
import SwiftData

@MainActor
final class RAGService: ObservableObject {
    static let shared = RAGService()

    @Published var isProcessing = false
    @Published var processingStatus: String = ""
    @Published var lastError: String?

    private let vectorDB = VectorDBService.shared

    private init() {}

    /// Streaming events emitted while answering a query.
    enum RAGEvent {
        case sources([ChatContext])
        case token(String)
    }

    enum RAGError: LocalizedError {
        case modelsNotReady
        var errorDescription: String? {
            switch self {
            case .modelsNotReady: return "The AI model isn't ready. Check Settings → AI Mode."
            }
        }
    }

    // MARK: - Wiring

    /// No-op: retrieval (VectorDBService/LocalVectorStore) needs no SwiftData
    /// context. Kept so ContentView's launch-time call site doesn't need to change.
    func configure(modelContext: ModelContext) {}

    // MARK: - Readiness

    /// Whether the active AI mode's provider is ready to generate (BYOK: a key
    /// is saved; On-Device: a model is downloaded and selected).
    var isReady: Bool {
        switch Configuration.llmMode {
        case .openAI:
            return !((try? KeychainService.shared.retrieveAPIKey(for: "openai")) ?? "").isEmpty
        case .localLLM:
            let path = Configuration.BYOK.localModelPath
            guard !path.isEmpty else { return false }
            return FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: path).appendingPathComponent("config.json").path)
        }
    }

    // MARK: - Indexing

    func embedJournalEntry(_ entry: JournalEntry) async {
        isProcessing = true
        processingStatus = "Indexing entry…"
        do {
            _ = try await index(
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
        isProcessing = true
        processingStatus = "Indexing profile item…"
        do {
            _ = try await index(
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
        vectorDB.deleteSource(entry.id.uuidString)
    }

    func deleteProfileItem(_ item: ProfileItem) async {
        vectorDB.deleteSource(item.id.uuidString)
    }

    /// Generic indexing entry point used by Email/Drive ingestion. Splits
    /// `fullText` into overlapping chunks and embeds each one. Returns the
    /// number of chunks stored.
    @discardableResult
    func index(
        sourceId: String,
        type: String,
        title: String,
        fullText: String,
        subtitle: String = "",
        sourceDate: Date? = nil
    ) async throws -> Int {
        vectorDB.deleteSource(sourceId)

        let chunks = TextChunker.chunk(fullText)
        guard !chunks.isEmpty else { return 0 }

        for (i, chunkText) in chunks.enumerated() {
            try await vectorDB.upsert(
                id: "\(sourceId)#\(i)",
                content: chunkText,
                type: type,
                title: title,
                subtitle: subtitle,
                sourceDate: sourceDate
            )
        }
        return chunks.count
    }

    /// Remove all chunks for a source (journal/profile/email/document).
    func removeSource(_ sourceId: String) {
        vectorDB.deleteSource(sourceId)
    }

    /// Wipe the entire on-device index (used by "Clear AI Data").
    func clearIndex() {
        vectorDB.deleteAll()
    }

    // MARK: - Retrieval + Generation (streaming)

    /// Stream an answer: first a `.sources` event with the retrieved citations,
    /// then `.token` events as the model generates.
    func streamResponse(
        to query: String,
        history: [(role: String, content: String)] = []
    ) -> AsyncThrowingStream<RAGEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard isReady else {
                    continuation.finish(throwing: RAGError.modelsNotReady); return
                }
                do {
                    let hits = try await vectorDB.search(query: query)
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
                    let providerHistory = history.map { ["role": $0.role, "content": $0.content] }

                    let stream = LLMService.activeProvider.streamComplete(
                        systemPrompt: Configuration.LLM.systemPrompt,
                        userMessage: userMessage,
                        history: providerHistory
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
        guard !entries.isEmpty, isReady else { return nil }
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
            return try await LLMService.activeProvider.complete(
                systemPrompt: Configuration.LLM.systemPrompt,
                userMessage: prompt,
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
        hits.prefix(Configuration.RAG.maxContextChunks).enumerated().map { index, hit in
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
