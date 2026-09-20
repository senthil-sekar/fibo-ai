//
//  ModelManager.swift
//  MindVault
//
//  Owns the on-device model lifecycle: download + load progress, readiness,
//  and model switching. The single source of truth the UI observes before
//  enabling chat / indexing.
//

import Foundation
import Combine

@MainActor
final class ModelManager: ObservableObject {
    static let shared = ModelManager()

    // MARK: - Published state

    enum Phase: Equatable {
        case idle               // nothing loaded yet
        case downloading        // fetching weights from HF Hub (first run)
        case loading            // weights present, loading into memory
        case ready              // both engines usable
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    /// 0...1 download progress for whichever model is currently fetching.
    @Published private(set) var downloadProgress: Double = 0
    @Published private(set) var statusMessage: String = ""
    @Published private(set) var activeLLM: LLMModelOption = OnDeviceConfig.selectedLLMModel

    var isReady: Bool { phase == .ready }

    // MARK: - Engines

    private(set) var llm: LLMEngine
    private(set) var embedder: EmbeddingEngine

    private init() {
        self.llm = LLMEngine(option: OnDeviceConfig.selectedLLMModel)
        self.embedder = EmbeddingEngine(option: OnDeviceConfig.selectedEmbeddingModel)
    }

    // MARK: - Lifecycle

    /// Load the embedding model (small) then the chat model (large).
    /// Safe to call repeatedly; it no-ops once ready.
    func prepare() async {
        guard phase != .ready else { return }
        do {
            phase = .downloading

            statusMessage = "Preparing \(OnDeviceConfig.selectedEmbeddingModel.displayName)…"
            try await embedder.load { [weak self] fraction in
                Task { @MainActor in self?.downloadProgress = fraction }
            }

            statusMessage = "Preparing \(activeLLM.displayName)…"
            downloadProgress = 0
            try await llm.load { [weak self] fraction in
                Task { @MainActor in
                    self?.downloadProgress = fraction
                    if fraction < 1 { self?.phase = .downloading }
                    else { self?.phase = .loading }
                }
            }

            statusMessage = ""
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
            statusMessage = error.localizedDescription
        }
    }

    /// Switch the chat model (from Settings). Triggers a fresh download/load.
    func switchLLM(to option: LLMModelOption) async {
        guard option != activeLLM else { return }
        OnDeviceConfig.selectedLLMModel = option
        activeLLM = option
        await llm.switchModel(to: option)
        phase = .idle
        downloadProgress = 0
        await prepare()
    }
}
