//
//  EmbeddingEngine.swift
//  MindVault
//
//  On-device text embeddings via MLXEmbedders (EmbeddingGemma by default).
//  Produces normalized vectors for the VectorStore. No network at inference time.
//

import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Generates embeddings entirely on-device.
///
/// An `actor` so all MLX work is serialized.
actor EmbeddingEngine {

    private var container: EmbedderModelContainer?
    private let option: EmbeddingModelOption

    /// Output dimensionality, detected from the first embedding (0 until then).
    private(set) var dimension: Int = 0

    var isLoaded: Bool { container != nil }

    init(option: EmbeddingModelOption) {
        self.option = option
    }

    // MARK: - Loading

    /// Downloads (first run) and loads the embedding model.
    /// `progress` is the download fraction in 0...1.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard container == nil else { return }
        container = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: option.huggingFaceRepo),
            progressHandler: { p in progress(p.fractionCompleted) }
        )
    }

    // MARK: - Embedding

    /// Embed a single string. `isQuery` selects the query vs. document prefix.
    func embed(_ text: String, isQuery: Bool) async throws -> [Float] {
        try await embed([text], isQuery: isQuery).first ?? []
    }

    /// Embed a batch of strings. Returns one normalized vector per input.
    func embed(_ texts: [String], isQuery: Bool) async throws -> [[Float]] {
        guard let container else { throw EmbeddingError.notLoaded }
        guard !texts.isEmpty else { return [] }

        let prefix = isQuery ? option.queryPrefix : option.documentPrefix
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)

        for text in texts {
            let prepared = prefix + text
            let vector: [Float] = await container.perform { context in
                let tokens = context.tokenizer.encode(text: prepared, addSpecialTokens: true)
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let output = context.model(
                    input, positionIds: nil, tokenTypeIds: nil, attentionMask: nil
                )
                let pooled = context.pooling(output, normalize: true, applyLayerNorm: false)
                pooled.eval()
                return pooled.asArray(Float.self)
            }
            vectors.append(vector)
        }

        if dimension == 0, let first = vectors.first { dimension = first.count }
        return vectors
    }

    enum EmbeddingError: LocalizedError {
        case notLoaded
        var errorDescription: String? {
            switch self {
            case .notLoaded: return "The embedding model has not finished loading yet."
            }
        }
    }
}
