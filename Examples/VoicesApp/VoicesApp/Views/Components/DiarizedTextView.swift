import SwiftUI

/// Renders diarized transcription segments as a single selectable Text
/// with per-speaker colors using SwiftUI Text concatenation.
struct DiarizedTextView: View {
    let segments: [DiarizedSegment]
    var font: Font = .body

    var body: some View {
        if segments.isEmpty {
            EmptyView()
        } else {
            segments.enumerated().reduce(Text("")) { result, pair in
                let (index, segment) = pair
                let prefix = index > 0 ? Text(" ") : Text("")
                return result + prefix + Text(segment.text)
                    .foregroundColor(SpeakerColors.color(for: segment.speaker))
            }
            .font(font)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Shows colored dots with speaker labels for active speakers.
struct SpeakerLegend: View {
    let activeSpeakers: Set<Int>

    var body: some View {
        if activeSpeakers.isEmpty {
            EmptyView()
        } else {
            HStack(spacing: 12) {
                ForEach(Array(activeSpeakers).sorted(), id: \.self) { speaker in
                    HStack(spacing: 4) {
                        Circle()
                            .fill(SpeakerColors.color(for: speaker))
                            .frame(width: 8, height: 8)
                        Text(SpeakerColors.label(for: speaker))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}
