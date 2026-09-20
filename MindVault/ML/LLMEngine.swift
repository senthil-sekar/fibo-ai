//
//  LLMEngine.swift
//  MindVault
//
//  On-device chat generation with Gemma 3n via MLX Swift.
//  Streams tokens as they are produced. No network at inference time.
//

import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Loads a Gemma 3n model and streams chat completions via `ChatSession`.
///
/// An `actor` so the underlying model container is accessed in a controlled way.
actor LLMEngine {

    private var container: ModelContainer?
    private(set) var option: LLMModelOption

    var isLoaded: Bool { container != nil }

    init(option: LLMModelOption) {
        self.option = option
    }

    /// Maps our app-level option to the package's predefined Gemma 3n config.
    private static func configuration(for option: LLMModelOption) -> ModelConfiguration {
        switch option {
        case .gemma3n_E2B: return LLMRegistry.gemma3n_E2B_it_lm_4bit
        case .gemma3n_E4B: return LLMRegistry.gemma3n_E4B_it_lm_4bit
        }
    }

    // MARK: - Loading

    /// Downloads (first run) and loads the selected chat model.
    /// `progress` is the download fraction in 0...1.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard container == nil else { return }
        container = try await #huggingFaceLoadModelContainer(
            configuration: Self.configuration(for: option),
            progressHandler: { p in progress(p.fractionCompleted) }
        )
    }

    /// Switch to a different model (used by Settings). Frees the old container.
    func switchModel(to newOption: LLMModelOption) {
        guard newOption != option else { return }
        option = newOption
        container = nil
    }

    // MARK: - Generation

    /// Stream a chat completion as incremental text deltas.
    func generate(
        system: String,
        user: String,
        history: [(role: String, content: String)]
    ) -> AsyncThrowingStream<String, Error> {
        guard let container else {
            return AsyncThrowingStream { $0.finish(throwing: LLMError.notLoaded) }
        }

        let messages: [Chat.Message] = history.map { turn in
            turn.role == "assistant" ? .assistant(turn.content) : .user(turn.content)
        }

        let parameters = GenerateParameters(
            maxTokens: OnDeviceConfig.maxOutputTokens,
            temperature: OnDeviceConfig.temperature
        )

        // ChatSession prepends `system` as instructions and replays `history`,
        // then streams the response to `user`. It owns the KV cache internally.
        let session = ChatSession(
            container,
            instructions: system,
            history: messages,
            generateParameters: parameters
        )
        return session.streamResponse(to: user)
    }

    /// Collect the full streamed response into a single string.
    func generateComplete(
        system: String,
        user: String,
        history: [(role: String, content: String)]
    ) async throws -> String {
        var result = ""
        for try await delta in generate(system: system, user: user, history: history) {
            result += delta
        }
        return result
    }

    enum LLMError: LocalizedError {
        case notLoaded
        var errorDescription: String? {
            switch self {
            case .notLoaded: return "The chat model has not finished loading yet."
            }
        }
    }
}
