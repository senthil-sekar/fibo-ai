//
//  LLMEngine.swift
//  MindVault
//
//  On-device chat generation with Gemma 3n via MLX Swift.
//  Streams tokens as they are produced. No network at inference time.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Loads a Gemma 3n model and streams chat completions.
///
/// An `actor` so MLX generation is serialized; the public `generate` API hands
/// text back through an `AsyncThrowingStream` that the UI can render live.
actor LLMEngine {

    private var llmModel: LLMModel?
    private var tokenizer: Tokenizer?
    private(set) var option: LLMModelOption

    var isLoaded: Bool { llmModel != nil && tokenizer != nil }

    init(option: LLMModelOption) {
        self.option = option
    }

    // MARK: - Loading

    /// Downloads (first run) and loads the selected chat model.
    /// `progress` is the download fraction in 0...1.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard llmModel == nil else { return }
        
        // Load model from Hugging Face Hub
        let modelConfig = ModelConfiguration.configuration(id: option.huggingFaceRepo)
        
        let loadedData = try await LLM.load(configuration: modelConfig) { loadProgress in
            Task { @MainActor in
                progress(loadProgress.fractionCompleted)
            }
        }
        
        self.llmModel = loadedData.model
        self.tokenizer = loadedData.tokenizer
    }

    /// Switch to a different model (used by the Settings picker). Frees the old
    /// container so memory is reclaimed before the new one loads.
    func switchModel(to newOption: LLMModelOption) {
        guard newOption != option else { return }
        option = newOption
        llmModel = nil
        tokenizer = nil
    }

    // MARK: - Generation

    /// Stream a chat completion. Yields incremental text deltas.
    /// - Parameters:
    ///   - system: system prompt (rules + persona).
    ///   - user: the user's message, already augmented with retrieved context.
    ///   - history: prior turns as (role, content) where role is "user"/"assistant".
    func generate(
        system: String,
        user: String,
        history: [(role: String, content: String)]
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let llmModel = self.llmModel, let tokenizer = self.tokenizer else {
                    continuation.finish(throwing: LLMError.notLoaded)
                    return
                }

                // Assemble the chat transcript
                var fullPrompt = system + "\n\n"
                for turn in history {
                    if turn.role == "user" {
                        fullPrompt += "User: \(turn.content)\n\n"
                    } else {
                        fullPrompt += "Assistant: \(turn.content)\n\n"
                    }
                }
                fullPrompt += "User: \(user)\n\nAssistant:"

                do {
                    // Tokenize the prompt
                    let tokens = tokenizer.encode(text: fullPrompt)
                    let promptTokens = MLXArray(tokens)
                    
                    let maxTokens = OnDeviceConfig.maxOutputTokens
                    var generatedTokens: [Int] = []
                    var currentTokens = promptTokens
                    
                    // Generate tokens one at a time
                    for _ in 0..<maxTokens {
                        // Add batch dimension
                        let input = currentTokens.expandedDimensions(axis: 0)
                        
                        // Forward pass
                        let logits = llmModel(input)
                        
                        // Get logits for last token
                        let lastLogits = logits[0, -1]
                        
                        // Apply temperature
                        let scaledLogits = lastLogits / OnDeviceConfig.temperature
                        
                        // Sample next token (argmax for now, could use sampling)
                        eval(scaledLogits)
                        let nextToken = argMax(scaledLogits).item(Int.self)
                        
                        // Check for end of sequence
                        if nextToken == tokenizer.unknownTokenId || nextToken == tokenizer.eosTokenId {
                            break
                        }
                        
                        generatedTokens.append(nextToken)
                        
                        // Decode and yield the new token
                        let decoded = tokenizer.decode(tokens: generatedTokens)
                        continuation.yield(decoded)
                        
                        // Append to current sequence
                        let nextTokenArray = MLXArray([nextToken])
                        currentTokens = concatenated([currentTokens, nextTokenArray], axis: 0)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Convenience: collect the full streamed response into a single string.
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
