import SwiftUI

/// A segment of transcription text attributed to a speaker.
struct DiarizedSegment: Identifiable {
    let id = UUID()
    var speaker: Int?  // 0-3, or nil for unknown/unassigned
    var text: String
}

/// Color palette for up to 4 speakers.
enum SpeakerColors {
    static let colors: [Color] = [
        .blue,    // Speaker 0
        .orange,  // Speaker 1
        .green,   // Speaker 2
        .purple   // Speaker 3
    ]

    static func color(for speaker: Int?) -> Color {
        guard let speaker, speaker >= 0, speaker < colors.count else {
            return .primary
        }
        return colors[speaker]
    }

    static func label(for speaker: Int?) -> String {
        guard let speaker else { return "Unknown" }
        return "Speaker \(speaker + 1)"
    }
}
