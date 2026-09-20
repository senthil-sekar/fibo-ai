//
//  EmbeddingService.swift
//  Fibo
//
//  On-device text embeddings via NLEmbedding — no backend, no download, no network.
//

import Foundation
import NaturalLanguage

enum EmbeddingService {
    /// NLEmbedding.sentenceEmbedding — built-in on iOS, ~512-dim.
    static func embedLocally(_ text: String) throws -> [Float] {
        guard let model = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw EmbeddingError.modelUnavailable
        }
        guard let vector = model.vector(for: text) else {
            throw EmbeddingError.failedToEmbed
        }
        return vector.map { Float($0) }
    }
}

// MARK: - Embedding Errors

enum EmbeddingError: LocalizedError {
    case modelUnavailable
    case failedToEmbed

    var errorDescription: String? {
        switch self {
        case .modelUnavailable:
            #if targetEnvironment(simulator)
            return "NLEmbedding's sentence embedding model isn't available in the iOS Simulator "
                + "(a known platform limitation, not a bug) — test on a physical device instead."
            #else
            return "On-device sentence embedding model is not available on this device."
            #endif
        case .failedToEmbed:
            return "Failed to generate embedding for the provided text."
        }
    }
}
