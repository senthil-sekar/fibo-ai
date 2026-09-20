//
//  TextChunker.swift
//  Fibo
//
//  Splits long text into overlapping, sentence-aware chunks before embedding.
//

import Foundation

enum TextChunker {
    /// Split text into overlapping, sentence-aware chunks. Short text returns a
    /// single chunk.
    static func chunk(
        _ text: String,
        size: Int = Configuration.RAG.chunkSize,
        overlap: Int = Configuration.RAG.chunkOverlap
    ) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard trimmed.count > size else { return [trimmed] }

        let chars = Array(trimmed)
        var chunks: [String] = []
        var start = 0

        while start < chars.count {
            var end = min(start + size, chars.count)

            // Try to end on a sentence/whitespace boundary for cleaner chunks.
            if end < chars.count {
                let windowStart = max(start + size - overlap, start)
                if let boundary = (windowStart..<end).reversed().first(where: {
                    let c = chars[$0]
                    return c == "." || c == "\n" || c == "!" || c == "?"
                }) {
                    end = boundary + 1
                }
            }

            let piece = String(chars[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { chunks.append(piece) }

            if end >= chars.count { break }
            start = max(end - overlap, start + 1)
        }
        return chunks
    }
}
