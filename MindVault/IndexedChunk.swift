//
//  IndexedChunk.swift
//  MindVault
//
//  SwiftData record for one embedded chunk of content. This is both the
//  vector store row and the metadata sidecar: it carries the embedding plus
//  everything the chat citations and source filters need.
//

import Foundation
import SwiftData

@Model
final class IndexedChunk {
    /// Unique chunk id, e.g. "<sourceId>#<chunkIndex>".
    @Attribute(.unique) var id: String

    /// Id of the parent item (journal entry / profile item / email / document).
    /// All chunks of a source share this, so deletes can remove them together.
    var sourceId: String

    /// "journal" | "profile" | "email" | "document"
    var type: String

    /// Human-facing title used in citations (entry title, email subject, filename).
    var title: String

    /// The chunk text that was embedded (also shown as the citation snippet).
    var text: String

    /// Source-specific subtitle, e.g. email sender or document folder.
    var subtitle: String

    /// Original date of the source item (journal date, email date, modified time).
    var sourceDate: Date?

    /// Normalized embedding vector, stored as packed little-endian Float32.
    var vectorData: Data

    /// Embedding dimensionality (lets us skip rows from a different model).
    var dimension: Int

    var createdAt: Date

    init(
        id: String,
        sourceId: String,
        type: String,
        title: String,
        text: String,
        subtitle: String = "",
        sourceDate: Date? = nil,
        vector: [Float],
        createdAt: Date = Date()
    ) {
        self.id = id
        self.sourceId = sourceId
        self.type = type
        self.title = title
        self.text = text
        self.subtitle = subtitle
        self.sourceDate = sourceDate
        self.vectorData = IndexedChunk.pack(vector)
        self.dimension = vector.count
        self.createdAt = createdAt
    }

    /// The embedding as `[Float]`.
    var vector: [Float] {
        IndexedChunk.unpack(vectorData, count: dimension)
    }

    // MARK: - Packing helpers

    static func pack(_ vector: [Float]) -> Data {
        vector.withUnsafeBytes { Data($0) }
    }

    static func unpack(_ data: Data, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self).prefix(count))
        }
    }
}
