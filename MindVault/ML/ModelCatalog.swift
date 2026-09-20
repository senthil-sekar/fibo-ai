//
//  ModelCatalog.swift
//  MindVault
//
//  Catalog of on-device models (LLM + embeddings) and RAG configuration.
//  Everything runs locally via MLX Swift — no backend, no network for inference
//  (only the initial weight download from the Hugging Face Hub).
//

import Foundation

// MARK: - LLM Model Options

/// Selectable on-device chat models. Gemma 3n is purpose-built for mobile
/// (Per-Layer Embeddings → smaller memory footprint than its parameter count
/// suggests). The text-only ("-lm") MLX builds are used since MindVault is
/// text-only for now.
enum LLMModelOption: String, CaseIterable, Identifiable, Codable {
    case gemma3n_E2B
    case gemma3n_E4B

    var id: String { rawValue }

    /// Hugging Face repo id for the 4-bit, text-only MLX build.
    /// NOTE: verify these repo ids resolve in your MLX/HF setup; the
    /// mlx-community naming for Gemma 3n text builds is `gemma-3n-E*B-it-lm-*bit`.
    var huggingFaceRepo: String {
        switch self {
        case .gemma3n_E2B: return "mlx-community/gemma-3n-E2B-it-lm-4bit"
        case .gemma3n_E4B: return "mlx-community/gemma-3n-E4B-it-lm-4bit"
        }
    }

    var displayName: String {
        switch self {
        case .gemma3n_E2B: return "Gemma 3n E2B"
        case .gemma3n_E4B: return "Gemma 3n E4B"
        }
    }

    var subtitle: String {
        switch self {
        case .gemma3n_E2B: return "Faster · lower memory · iPhone 15 Pro / 16+"
        case .gemma3n_E4B: return "Higher quality · needs the most RAM / newest devices"
        }
    }

    /// Rough on-disk download size, for the UI.
    var approxDownloadSize: String {
        switch self {
        case .gemma3n_E2B: return "~2.5 GB"
        case .gemma3n_E4B: return "~4.5 GB"
        }
    }
}

// MARK: - Embedding Model Options

/// Selectable on-device embedding models, all supported by MLXEmbedders
/// (EmbeddingGemma via the `gemma3` architecture; Nomic via `nomic_bert`; BGE via `bert`).
///
/// The vector store detects dimensionality at runtime, so switching models only
/// requires re-indexing — no other code changes.
enum EmbeddingModelOption: String, CaseIterable, Identifiable, Codable {
    case embeddingGemma
    case nomicTextV15
    case bgeSmallEN

    var id: String { rawValue }

    var huggingFaceRepo: String {
        switch self {
        case .embeddingGemma: return "mlx-community/embeddinggemma-300m-bf16"
        case .nomicTextV15:   return "nomic-ai/nomic-embed-text-v1.5"
        case .bgeSmallEN:     return "BAAI/bge-small-en-v1.5"
        }
    }

    var displayName: String {
        switch self {
        case .embeddingGemma: return "EmbeddingGemma 300M"
        case .nomicTextV15:   return "Nomic Embed Text v1.5"
        case .bgeSmallEN:     return "BGE Small EN v1.5"
        }
    }

    /// Instruction-tuned models embed queries and documents with different
    /// task prefixes; applying the right one improves retrieval quality.
    /// (EmbeddingGemma's prompt format per Google's model card.)
    var queryPrefix: String {
        switch self {
        case .embeddingGemma: return "task: search result | query: "
        case .nomicTextV15:   return "search_query: "
        case .bgeSmallEN:     return "Represent this sentence for searching relevant passages: "
        }
    }

    var documentPrefix: String {
        switch self {
        case .embeddingGemma: return "title: none | text: "
        case .nomicTextV15:   return "search_document: "
        case .bgeSmallEN:     return ""
        }
    }
}

// MARK: - On-Device Configuration

/// Central knobs for the on-device RAG pipeline and persisted user choices.
enum OnDeviceConfig {

    // MARK: Persisted selections

    private enum Keys {
        static let llmModel = "selectedLLMModel"
        static let embeddingModel = "selectedEmbeddingModel"
    }

    static var selectedLLMModel: LLMModelOption {
        get {
            guard let raw = UserDefaults.standard.string(forKey: Keys.llmModel),
                  let option = LLMModelOption(rawValue: raw) else {
                return .gemma3n_E2B   // sensible default for the broadest device range
            }
            return option
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: Keys.llmModel) }
    }

    static var selectedEmbeddingModel: EmbeddingModelOption {
        get {
            guard let raw = UserDefaults.standard.string(forKey: Keys.embeddingModel),
                  let option = EmbeddingModelOption(rawValue: raw) else {
                return .embeddingGemma
            }
            return option
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: Keys.embeddingModel) }
    }

    // MARK: Retrieval

    /// How many chunks to retrieve for a query.
    static let topK = 6
    /// Minimum cosine similarity for a chunk to be considered relevant.
    /// Kept low so broad "summarize my emails" style queries still surface context.
    static let minRelevanceScore: Float = 0.15
    /// Cap on chunks fed into the prompt (protects the context window).
    static let maxContextChunks = 8

    // MARK: Chunking (for long emails / documents)

    /// Target characters per chunk. ~1200 chars ≈ a few hundred tokens.
    static let chunkSize = 1200
    /// Overlap between consecutive chunks to preserve context across boundaries.
    static let chunkOverlap = 150

    // MARK: Generation

    static let maxOutputTokens = 800
    static let temperature: Float = 0.7

    // MARK: System prompt

    static let systemPrompt = """
    You are MindVault, a personal AI assistant that runs entirely on the user's \
    iPhone. You answer questions using ONLY the CONTEXT retrieved from the user's \
    own journal entries, profile, emails, and documents.

    Rules:
    1. Use only the provided context. Never invent journal entries, emails, or facts.
    2. Cite specifics: the journal date, the email sender/subject, or the document name.
    3. If the context doesn't contain the answer, say so honestly and briefly.
    4. Be warm, concise, and specific. Use bullet points when listing multiple items.
    """
}
