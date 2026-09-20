//
//  EmbeddingEngine.swift
//  MindVault
//
//  On-device text embeddings via MLXEmbedders. Produces normalized vectors
//  used by the VectorStore for semantic retrieval. No network at inference time.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Generates embeddings entirely on-device.
///
/// An `actor` so all MLX work is serialized (MLX evaluation is not meant to be
/// driven concurrently from multiple tasks).
actor EmbeddingEngine {

    private var model: LLMModel?
    private var tokenizer: Tokenizer?
    private let option: EmbeddingModelOption

    /// Dimensionality of the loaded model's output, detected from the first
    /// embedding. 0 until the first `embed` completes.
    private(set) var dimension: Int = 0

    var isLoaded: Bool { model != nil && tokenizer != nil }

    init(option: EmbeddingModelOption) {
        self.option = option
    }

    // MARK: - Loading

    /// Downloads (first run) and loads the embedding model.
    /// `progress` is the download fraction in 0...1.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard model == nil else { return }
        
        // Load model from Hugging Face Hub
        let modelConfig = ModelConfiguration.configuration(id: option.huggingFaceRepo)
        
        let loadedData = try await LLM.load(configuration: modelConfig) { loadProgress in
            Task { @MainActor in
                progress(loadProgress.fractionCompleted)
            }
        }
        
        self.model = loadedData.model
        self.tokenizer = loadedData.tokenizer
    }

    // MARK: - Embedding

    /// Embed a single string. `isQuery` selects the model's query vs. document
    /// instruction prefix (improves retrieval quality for instruction-tuned models).
    func embed(_ text: String, isQuery: Bool) async throws -> [Float] {
        let results = try await embed([text], isQuery: isQuery)
        return results.first ?? []
    }

    /// Embed a batch of strings. Returns one normalized vector per input.
    func embed(_ texts: [String], isQuery: Bool) async throws -> [[Float]] {
        guard let model = self.model, let tokenizer = self.tokenizer else {
            throw EmbeddingError.notLoaded
        }
        guard !texts.isEmpty else { return [] }

        let prefix = isQuery ? option.queryPrefix : option.documentPrefix
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)

        // Embed one at a time: simple and robust at personal-corpus scale.
        for text in texts {
            let prepared = prefix + text
            
            // Tokenize
            let tokens = tokenizer.encode(text: prepared)
            let inputIds = MLXArray(tokens)
            let batch = inputIds.expandedDimensions(axis: 0)
            
            // Run through model to get hidden states
            let output = model(batch)
            
            // Extract last hidden state (embedding)
            // Most embedding models output a dictionary with "last_hidden_state"
            let hiddenStates: MLXArray
            if let dict = output as? [String: MLXArray], let lastHidden = dict["last_hidden_state"] {
                hiddenStates = lastHidden
            } else if let array = output as? MLXArray {
                hiddenStates = array
            } else {
                throw EmbeddingError.invalidOutput
            }
            
            // Mean pooling over sequence dimension (axis 1)
            let pooled = mean(hiddenStates, axis: 1)
            
            // L2 normalize
            let norm = sqrt(sum(pooled * pooled, axis: -1, keepDims: true))
            let normalized = pooled / (norm + 1e-8)  // Add epsilon to avoid division by zero
            
            // Evaluate and convert to array
            eval(normalized)
            let vector = normalized.squeezed().asArray(Float.self)
            vectors.append(vector)
        }

        if dimension == 0, let first = vectors.first { dimension = first.count }
        return vectors
    }

    enum EmbeddingError: LocalizedError {
        case notLoaded
        case invalidOutput
        
        var errorDescription: String? {
            switch self {
            case .notLoaded:
                return "The embedding model has not finished loading yet."
            case .invalidOutput:
                return "The model produced an unexpected output format."
            }
        }
    }
}
