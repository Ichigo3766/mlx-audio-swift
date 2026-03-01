import Foundation
import SwiftUI
import MLXAudioSTT
import MLXAudioCore
import MLXAudioVAD
import MLX
@preconcurrency import AVFoundation
import Combine

@MainActor
@Observable
class STTViewModel {
    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var transcriptionText: String = ""
    var tokensPerSecond: Double = 0
    var peakMemory: Double = 0

    // Generation parameters
    var maxTokens: Int = 1024
    var temperature: Float = 0.0
    var language: String = "English"
    var chunkDuration: Float = 30.0

    // Streaming parameters
    var streamingDelayMs: Int = 480  // .agent default

    // Model configuration
    var modelId: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private var loadedModelId: String?

    // Diarization parameters
    var isDiarizationEnabled: Bool = false
    var diarizationThreshold: Float = 0.5
    var diarizedSegments: [DiarizedSegment] = []
    var diarizationModelId: String = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
    var isDiarizationModelLoaded: Bool { diarizationModel != nil }

    // Audio file
    var selectedAudioURL: URL?
    var audioFileName: String?

    // Audio player state
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    // Recording state
    var isRecording: Bool { recorder.isRecording }
    var recordingDuration: TimeInterval { recorder.recordingDuration }
    var audioLevel: Float { recorder.audioLevel }

    private var model: Qwen3ASRModel?
    private let audioPlayer = AudioPlayer()
    private let recorder = AudioRecorderManager()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

    // Diarization state
    private var diarizationModel: SortformerModel?
    private var diarizationLoadedModelId: String?
    private var diarizationStreamingState: StreamingState?
    private var currentSpeaker: Int?
    private var diarLiveTask: Task<Void, Never>?
    private var diarLastReadPos: Int = 0
    private var confirmedDiarSegments: [DiarizedSegment] = []
    private var lastConfirmedLength: Int = 0

    // Forced aligner (loaded on-demand for offline diarization)
    private var alignerModel: Qwen3ForcedAlignerModel?
    private let alignerModelId = "mlx-community/Qwen3-ForcedAligner-0.6B-4bit"

    var isModelLoaded: Bool {
        model != nil
    }

    /// Set of active speaker IDs from current diarized segments.
    var activeSpeakers: Set<Int> {
        Set(diarizedSegments.compactMap(\.speaker))
    }

    init() {
        setupAudioPlayerObservers()
    }

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isPlaying = value
            }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.currentTime = value
            }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.duration = value
            }
            .store(in: &cancellables)
    }

    func loadModel() async {
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            model = try await Qwen3ASRModel.fromPretrained(modelId)
            loadedModelId = modelId
            generationProgress = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false

        // Load diarization model if enabled
        if isDiarizationEnabled {
            await loadDiarizationModel()
        }
    }

    func loadDiarizationModel() async {
        guard isDiarizationEnabled else { return }
        guard diarizationModel == nil || diarizationLoadedModelId != diarizationModelId else { return }

        generationProgress = "Downloading diarization model..."

        do {
            diarizationModel = try await SortformerModel.fromPretrained(diarizationModelId)
            diarizationLoadedModelId = diarizationModelId
            generationProgress = ""
        } catch {
            errorMessage = "Failed to load diarization model: \(error.localizedDescription)"
            generationProgress = ""
        }
    }

    func reloadModel() async {
        model = nil
        loadedModelId = nil
        diarizationModel = nil
        diarizationLoadedModelId = nil
        alignerModel = nil
        Memory.clearCache()
        await loadModel()
    }

    func selectAudioFile(_ url: URL) {
        selectedAudioURL = url
        audioFileName = url.lastPathComponent
        audioPlayer.loadAudio(from: url)
    }

    func startTranscription() {
        guard let audioURL = selectedAudioURL else {
            errorMessage = "No audio file selected"
            return
        }

        generationTask = Task {
            await transcribe(audioURL: audioURL)
        }
    }

    func transcribe(audioURL: URL) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        transcriptionText = ""
        diarizedSegments = []
        generationProgress = "Loading audio..."
        tokensPerSecond = 0
        peakMemory = 0

        do {
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
            let targetRate = model.sampleRate

            let resampled: MLXArray
            if sampleRate != targetRate {
                generationProgress = "Resampling \(sampleRate)Hz → \(targetRate)Hz..."
                resampled = try resampleAudio(audioData, from: sampleRate, to: targetRate)
            } else {
                resampled = audioData
            }

            if isDiarizationEnabled, let diarModel = diarizationModel {
                try await transcribeWithDiarization(audio: resampled, model: model, diarModel: diarModel)
            } else {
                try await transcribeSTTOnly(audio: resampled, model: model)
            }

            generationProgress = ""
        } catch is CancellationError {
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    private func transcribeSTTOnly(audio: MLXArray, model: Qwen3ASRModel) async throws {
        generationProgress = "Transcribing..."

        var tokenCount = 0
        for try await event in model.generateStream(
            audio: audio,
            maxTokens: maxTokens,
            temperature: temperature,
            language: language,
            chunkDuration: chunkDuration
        ) {
            try Task.checkCancellation()

            switch event {
            case .token(let token):
                transcriptionText += token
                tokenCount += 1
                generationProgress = "Transcribing... \(tokenCount) tokens"
            case .info(let info):
                tokensPerSecond = info.tokensPerSecond
                peakMemory = info.peakMemoryUsage
            case .result:
                generationProgress = ""
            }
        }
    }

    private func transcribeWithDiarization(
        audio: MLXArray,
        model: Qwen3ASRModel,
        diarModel: SortformerModel
    ) async throws {
        // Phase 1: Streaming diarization — matching Python pattern:
        //   state = model.init_streaming_state()
        //   for chunk in chunks:
        //       for result in model.generate_stream(chunk, state=state):
        //           state = result.state
        generationProgress = "Analyzing speakers..."
        let threshold = diarizationThreshold
        let sampleRate = model.sampleRate
        let chunkSize = 5 * sampleRate  // 5 seconds per chunk

        var allDiarSegments: [DiarizationSegment] = []
        var state = diarModel.initStreamingState()

        // nonisolated(unsafe) to avoid Swift 6 sending errors
        nonisolated(unsafe) let diarModelRef = diarModel

        let totalSamples = audio.dim(0)
        print("[Diar] Total samples: \(totalSamples), sampleRate: \(sampleRate), chunkSize: \(chunkSize), threshold: \(threshold)")
        var chunkIdx = 0
        for start in stride(from: 0, to: totalSamples, by: chunkSize) {
            try Task.checkCancellation()

            let end = min(start + chunkSize, totalSamples)
            nonisolated(unsafe) let chunk = audio[start..<end]

            let (result, newState) = try await diarModelRef.feed(
                chunk: chunk,
                state: state,
                sampleRate: sampleRate,
                threshold: threshold
            )
            state = newState

            print("[Diar] Chunk \(chunkIdx): \(String(format: "%.2f", Float(start)/Float(sampleRate)))s-\(String(format: "%.2f", Float(end)/Float(sampleRate)))s → \(result.segments.count) segments, state: spkcache=\(state.spkcacheLen) fifo=\(state.fifoLen) framesProcessed=\(state.framesProcessed)")
            for seg in result.segments {
                print("[Diar]   Speaker \(seg.speaker): \(String(format: "%.2f", seg.start))s - \(String(format: "%.2f", seg.end))s")
            }
            allDiarSegments.append(contentsOf: result.segments)
            chunkIdx += 1
        }

        print("[Diar] Total segments: \(allDiarSegments.count)")
        for seg in allDiarSegments {
            print("[Diar]   Speaker \(seg.speaker): \(String(format: "%.2f", seg.start))s - \(String(format: "%.2f", seg.end))s")
        }

        // Free diarization memory before STT
        Memory.clearCache()

        try Task.checkCancellation()

        // Phase 2: Transcribe full audio in one pass (best quality)
        generationProgress = "Transcribing..."
        var fullText = ""
        var tokenCount = 0

        for try await event in model.generateStream(
            audio: audio,
            maxTokens: maxTokens,
            temperature: temperature,
            language: language,
            chunkDuration: chunkDuration
        ) {
            try Task.checkCancellation()

            switch event {
            case .token(let token):
                fullText += token
                transcriptionText += token
                tokenCount += 1
                generationProgress = "Transcribing... \(tokenCount) tokens"
            case .info(let info):
                tokensPerSecond = info.tokensPerSecond
                peakMemory = info.peakMemoryUsage
            case .result:
                break
            }
        }

        fullText = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
        print("[STT] Full transcription (\(tokenCount) tokens): \"\(fullText.prefix(200))\"")

        guard !fullText.isEmpty, !allDiarSegments.isEmpty else {
            if !fullText.isEmpty {
                diarizedSegments = [DiarizedSegment(speaker: nil, text: fullText)]
            }
            return
        }

        Memory.clearCache()

        try Task.checkCancellation()

        // Phase 3: Forced alignment — get word-level timestamps
        generationProgress = "Aligning words to timestamps..."

        // Load aligner model on demand
        if alignerModel == nil {
            generationProgress = "Loading forced aligner model..."
            alignerModel = try await Qwen3ForcedAlignerModel.fromPretrained(alignerModelId)
        }

        guard let aligner = alignerModel else {
            // Fallback: no aligner available, show plain text
            diarizedSegments = [DiarizedSegment(speaker: nil, text: fullText)]
            return
        }

        generationProgress = "Aligning words..."
        let alignResult = aligner.generate(audio: audio, text: fullText, language: language)

        print("[Align] \(alignResult.items.count) words aligned in \(String(format: "%.2f", alignResult.totalTime))s")
        for item in alignResult.items {
            print("[Align]   [\(String(format: "%.2f", item.startTime))s-\(String(format: "%.2f", item.endTime))s] \"\(item.text)\"")
        }

        Memory.clearCache()

        // Phase 4: Map each word to its speaker using raw diarization segments,
        // then smooth at sentence boundaries (speaker changes don't happen mid-sentence).
        let originalWords = fullText.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        let alignItems = alignResult.items
        let wordCount = min(originalWords.count, alignItems.count)

        print("[Phase4] \(originalWords.count) original words, \(alignItems.count) aligned items, using \(wordCount)")

        // Step 1: Word-level speaker assignment
        struct WordAssignment {
            let word: String
            let speaker: Int?
        }
        var assignments: [WordAssignment] = []

        for i in 0..<wordCount {
            let item = alignItems[i]
            let word = originalWords[i]
            let speaker = Self.speakerForWord(
                startTime: Float(item.startTime),
                endTime: Float(item.endTime),
                segments: allDiarSegments
            )
            assignments.append(WordAssignment(word: word, speaker: speaker))
        }

        // Append any remaining original words not covered by aligner
        if originalWords.count > wordCount {
            let lastSpeaker = assignments.last?.speaker
            for i in wordCount..<originalWords.count {
                assignments.append(WordAssignment(word: originalWords[i], speaker: lastSpeaker))
            }
        }

        // Step 2: Group words into sentences (split at . ? !)
        var sentences: [[WordAssignment]] = []
        var currentSentence: [WordAssignment] = []

        for assignment in assignments {
            currentSentence.append(assignment)
            let w = assignment.word
            if w.hasSuffix(".") || w.hasSuffix("?") || w.hasSuffix("!") {
                sentences.append(currentSentence)
                currentSentence = []
            }
        }
        if !currentSentence.isEmpty {
            sentences.append(currentSentence)
        }

        // Step 3: For each sentence, majority-vote the speaker.
        // Speaker changes happen at sentence boundaries, not mid-sentence.
        var segments: [DiarizedSegment] = []

        for sentence in sentences {
            var speakerCounts: [Int: Int] = [:]
            for wa in sentence {
                if let spk = wa.speaker {
                    speakerCounts[spk, default: 0] += 1
                }
            }
            // Majority speaker; tie-break with first word's speaker
            let majoritySpeaker: Int?
            if let best = speakerCounts.max(by: { $0.value < $1.value }) {
                // Check for tie
                let maxCount = best.value
                let tied = speakerCounts.filter { $0.value == maxCount }
                if tied.count > 1, let firstSpk = sentence.first?.speaker {
                    majoritySpeaker = firstSpk
                } else {
                    majoritySpeaker = best.key
                }
            } else {
                majoritySpeaker = sentence.first?.speaker
            }

            let text = sentence.map(\.word).joined(separator: " ")

            print("[Phase4] Sentence (\(sentence.count)w, Spk \(majoritySpeaker.map(String.init) ?? "?")): \"\(text.prefix(80))\"")

            // Merge with previous segment if same speaker
            if let lastIdx = segments.indices.last,
               segments[lastIdx].speaker == majoritySpeaker {
                segments[lastIdx].text += " " + text
            } else {
                segments.append(DiarizedSegment(speaker: majoritySpeaker, text: text))
            }
        }

        diarizedSegments = segments
        transcriptionText = segments.map(\.text).joined(separator: " ")

        print("[Diar→STT] Final \(segments.count) display segments:")
        for seg in segments {
            let label = seg.speaker.map { "Speaker \($0)" } ?? "Unknown"
            print("[Diar→STT]   \(label): \"\(seg.text.prefix(120))\"")
        }
    }

    /// A non-overlapping speaker turn with start/end time.
    private struct SpeakerTurn {
        let start: Float
        let end: Float
        let speaker: Int
    }

    /// Build non-overlapping speaker turns that cover the full speech range.
    /// Samples the timeline at `frameStep` resolution, picks dominant speaker at each point,
    /// run-length encodes into turns, then fills gaps so no audio is lost.
    private static func buildSpeakerTurns(
        from segments: [DiarizationSegment],
        audioDuration: Float,
        frameStep: Float = 0.08
    ) -> [SpeakerTurn] {
        guard !segments.isEmpty else { return [] }

        // Sample timeline: at each frame, find the dominant speaker
        let numFrames = Int(audioDuration / frameStep) + 1
        var timeline: [Int?] = Array(repeating: nil, count: numFrames)

        for frame in 0..<numFrames {
            let time = Float(frame) * frameStep
            let active = segments.filter { time >= $0.start && time < $0.end }
            if active.isEmpty { continue }
            if active.count == 1 {
                timeline[frame] = active[0].speaker
            } else {
                // Overlap: pick speaker with longer segment
                var durations: [Int: Float] = [:]
                for seg in active {
                    durations[seg.speaker, default: 0] += (seg.end - seg.start)
                }
                timeline[frame] = durations.max(by: { $0.value < $1.value })?.key
            }
        }

        // Run-length encode into turns (only where a speaker is active)
        var rawTurns: [SpeakerTurn] = []
        var turnStart: Int? = nil
        var turnSpeaker: Int? = nil

        for frame in 0..<numFrames {
            let speaker = timeline[frame]
            if speaker != turnSpeaker {
                if let start = turnStart, let spk = turnSpeaker {
                    rawTurns.append(SpeakerTurn(
                        start: Float(start) * frameStep,
                        end: Float(frame) * frameStep,
                        speaker: spk
                    ))
                }
                turnStart = speaker != nil ? frame : nil
                turnSpeaker = speaker
            }
        }
        // Flush last turn
        if let start = turnStart, let spk = turnSpeaker {
            rawTurns.append(SpeakerTurn(
                start: Float(start) * frameStep,
                end: min(Float(numFrames) * frameStep, audioDuration),
                speaker: spk
            ))
        }

        guard !rawTurns.isEmpty else { return [] }

        // Merge consecutive same-speaker turns (across brief silence gaps < 1s)
        var merged: [SpeakerTurn] = []
        for turn in rawTurns {
            if let last = merged.last,
               last.speaker == turn.speaker,
               (turn.start - last.end) < 1.0 {
                merged[merged.count - 1] = SpeakerTurn(
                    start: last.start,
                    end: turn.end,
                    speaker: turn.speaker
                )
            } else {
                merged.append(turn)
            }
        }

        // Fill gaps between turns so no audio is lost:
        // - Gaps < 2s: split at midpoint between adjacent turns
        // - Gaps >= 2s: leave as-is (long silence, no speech to transcribe)
        var filled: [SpeakerTurn] = []
        for (i, turn) in merged.enumerated() {
            var adjustedStart = turn.start
            var adjustedEnd = turn.end

            if i > 0 {
                let prevEnd = filled[filled.count - 1].end
                let gap = turn.start - prevEnd
                if gap > 0 && gap < 2.0 {
                    // Split gap: extend previous turn to midpoint, start this turn from midpoint
                    let mid = prevEnd + gap / 2
                    filled[filled.count - 1] = SpeakerTurn(
                        start: filled[filled.count - 1].start,
                        end: mid,
                        speaker: filled[filled.count - 1].speaker
                    )
                    adjustedStart = mid
                }
            }

            if i < merged.count - 1 {
                let nextStart = merged[i + 1].start
                let gap = nextStart - turn.end
                if gap > 0 && gap < 2.0 {
                    // Extend this turn to midpoint (will be adjusted again by next iteration)
                    adjustedEnd = turn.end + gap / 2
                }
            }

            filled.append(SpeakerTurn(
                start: adjustedStart,
                end: adjustedEnd,
                speaker: turn.speaker
            ))
        }

        return filled
    }

    // MARK: - Live Recording & Streaming Transcription

    private var liveTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?
    private var streamingSession: StreamingInferenceSession?
    private var lastReadPos: Int = 0

    func startRecording() async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        // Load diarization model on demand if needed
        if isDiarizationEnabled && diarizationModel == nil {
            await loadDiarizationModel()
        }

        errorMessage = nil
        transcriptionText = ""
        diarizedSegments = []
        confirmedDiarSegments = []
        lastConfirmedLength = 0
        currentSpeaker = nil
        tokensPerSecond = 0
        peakMemory = 0
        lastReadPos = 0
        diarLastReadPos = 0

        do {
            try await recorder.startRecording()
        } catch {
            errorMessage = error.localizedDescription
            return
        }

        // Create streaming session
        let config = StreamingConfig(
            decodeIntervalSeconds: 1.0,
            maxCachedWindows: 60,
            delayPreset: .custom(ms: streamingDelayMs),
            language: language,
            temperature: temperature,
            maxTokensPerPass: maxTokens
        )
        let session = StreamingInferenceSession(model: model, config: config)
        streamingSession = session

        // Listen to events from the session
        eventTask = Task {
            for await event in session.events {
                switch event {
                case .displayUpdate(let confirmed, let provisional):
                    transcriptionText = confirmed + provisional
                    if isDiarizationEnabled {
                        updateDiarizedSegments(confirmedText: confirmed, provisionalText: provisional)
                    }
                case .confirmed:
                    break  // displayUpdate handles the UI
                case .provisional:
                    break
                case .stats(let stats):
                    tokensPerSecond = stats.tokensPerSecond
                    peakMemory = stats.peakMemoryGB
                case .ended(let fullText):
                    transcriptionText = fullText
                    if isDiarizationEnabled {
                        finalizeDiarizedSegments(fullText: fullText)
                    }
                }
            }
            // Stream ended naturally — clean up
            streamingSession = nil
            eventTask = nil
        }

        // Audio feed loop: read new samples every 100ms and feed to session
        liveTask = Task {
            while !Task.isCancelled && recorder.isRecording {
                if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                    lastReadPos = endPos
                    let samples = audio.asArray(Float.self)
                    session.feedAudio(samples: samples)
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }

        // Diarization feed loop (parallel, independent read position)
        if isDiarizationEnabled, let diarModel = diarizationModel {
            diarizationStreamingState = diarModel.initStreamingState()

            diarLiveTask = Task {
                while !Task.isCancelled && recorder.isRecording {
                    guard let diarState = diarizationStreamingState else { break }

                    if let (audio, endPos) = recorder.getAudio(from: diarLastReadPos) {
                        diarLastReadPos = endPos
                        let samples = audio.asArray(Float.self)
                        let chunk = MLXArray(samples)

                        do {
                            let (output, newState) = try await diarModel.feed(
                                chunk: chunk,
                                state: diarState,
                                threshold: diarizationThreshold
                            )
                            diarizationStreamingState = newState

                            let dominant = Self.dominantSpeaker(from: output.segments)
                            if dominant != currentSpeaker {
                                currentSpeaker = dominant
                            }
                        } catch {
                            // Non-fatal: diarization failure shouldn't stop transcription
                        }
                    }
                    try? await Task.sleep(for: .milliseconds(100))
                }
            }
        }
    }

    func stopRecording() {
        liveTask?.cancel()
        liveTask = nil
        diarLiveTask?.cancel()
        diarLiveTask = nil

        _ = recorder.stopRecording()

        // Feed any remaining audio, then stop session
        if let session = streamingSession {
            if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                lastReadPos = endPos
                let samples = audio.asArray(Float.self)
                session.feedAudio(samples: samples)
            }

            // Stop promotes all provisional tokens and emits .ended
            // The eventTask will process .ended and clean up naturally
            session.stop()
        }

        diarizationStreamingState = nil
    }

    func cancelRecording() {
        liveTask?.cancel()
        liveTask = nil
        diarLiveTask?.cancel()
        diarLiveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        recorder.cancelRecording()
        lastReadPos = 0
        diarLastReadPos = 0
        diarizationStreamingState = nil
        currentSpeaker = nil
    }

    func stop() {
        liveTask?.cancel()
        liveTask = nil
        diarLiveTask?.cancel()
        diarLiveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        generationTask?.cancel()
        generationTask = nil
        diarizationStreamingState = nil
        currentSpeaker = nil

        if isRecording {
            recorder.cancelRecording()
            lastReadPos = 0
            diarLastReadPos = 0
        }

        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    func play() {
        audioPlayer.play()
    }

    func pause() {
        audioPlayer.pause()
    }

    func togglePlayPause() {
        audioPlayer.togglePlayPause()
    }

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }

    func copyTranscription() {
        let text: String
        if isDiarizationEnabled && !diarizedSegments.isEmpty {
            text = diarizedSegments.map { seg in
                let label = SpeakerColors.label(for: seg.speaker)
                return "[\(label)]: \(seg.text)"
            }.joined(separator: "\n")
        } else {
            text = transcriptionText
        }

        #if os(iOS)
        UIPasteboard.general.string = text
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }

    // MARK: - Diarization Helpers

    /// Determine which speaker a word belongs to using raw diarization segments.
    /// Uses the word's full timespan for overlap calculation — more precise than
    /// quantized speaker turns.
    private static func speakerForWord(
        startTime: Float,
        endTime: Float,
        segments: [DiarizationSegment]
    ) -> Int? {
        let midTime = (startTime + endTime) / 2.0

        // Find all segments covering this word's midpoint
        let active = segments.filter { midTime >= $0.start && midTime < $0.end }

        if active.count == 1 {
            return active[0].speaker
        }

        if active.count > 1 {
            // Overlap zone: pick the most recently started segment.
            // When diarization segments overlap, the newer segment typically
            // represents the actual speaker change — the older segment's tail
            // is the model being slow to detect the transition ended.
            return active.max(by: { $0.start < $1.start })?.speaker
        }

        // No segment covers this word — find the nearest segment
        var minDist: Float = .infinity
        var nearest: Int? = nil
        for seg in segments {
            let dist = min(abs(midTime - seg.start), abs(midTime - seg.end))
            if dist < minDist {
                minDist = dist
                nearest = seg.speaker
            }
        }
        return nearest
    }

    /// Find which speaker is active at a given timestamp by checking speaker turns.
    /// Falls back to nearest turn if the time falls in a gap.
    private static func speakerAtTime(_ time: Float, turns: [SpeakerTurn]) -> Int? {
        // Direct hit: time falls within a turn
        for turn in turns {
            if time >= turn.start && time < turn.end {
                return turn.speaker
            }
        }
        // Fallback: find nearest turn
        var minDist: Float = .infinity
        var nearest: Int? = nil
        for turn in turns {
            let dist = min(abs(time - turn.start), abs(time - turn.end))
            if dist < minDist {
                minDist = dist
                nearest = turn.speaker
            }
        }
        return nearest
    }

    /// Find the speaker with the most total duration in the given segments.
    static func dominantSpeaker(from segments: [DiarizationSegment]) -> Int? {
        guard !segments.isEmpty else { return nil }

        var durations: [Int: Float] = [:]
        for seg in segments {
            durations[seg.speaker, default: 0] += (seg.end - seg.start)
        }
        return durations.max(by: { $0.value < $1.value })?.key
    }

    /// Update diarized segments during realtime streaming.
    /// Confirmed segments are frozen; provisional text is appended as mutable tail.
    private func updateDiarizedSegments(confirmedText: String, provisionalText: String) {
        let confirmedLen = confirmedText.count
        if confirmedLen > lastConfirmedLength {
            let newConfirmed = String(confirmedText.dropFirst(lastConfirmedLength))
            if !newConfirmed.isEmpty {
                if let lastIdx = confirmedDiarSegments.indices.last,
                   confirmedDiarSegments[lastIdx].speaker == currentSpeaker {
                    confirmedDiarSegments[lastIdx].text += newConfirmed
                } else {
                    confirmedDiarSegments.append(DiarizedSegment(
                        speaker: currentSpeaker, text: newConfirmed
                    ))
                }
            }
            lastConfirmedLength = confirmedLen
        }

        // Build display: confirmed segments + provisional tail
        var display = confirmedDiarSegments
        if !provisionalText.isEmpty {
            if let lastIdx = display.indices.last,
               display[lastIdx].speaker == currentSpeaker {
                // Copy last segment and append provisional text
                display[display.count - 1].text += provisionalText
            } else {
                display.append(DiarizedSegment(speaker: currentSpeaker, text: provisionalText))
            }
        }
        diarizedSegments = display
    }

    /// Finalize diarized segments when streaming ends.
    private func finalizeDiarizedSegments(fullText: String) {
        // If there's remaining text beyond what was confirmed, add it
        if fullText.count > lastConfirmedLength {
            let remaining = String(fullText.dropFirst(lastConfirmedLength))
            if !remaining.isEmpty {
                if let lastIdx = confirmedDiarSegments.indices.last,
                   confirmedDiarSegments[lastIdx].speaker == currentSpeaker {
                    confirmedDiarSegments[lastIdx].text += remaining
                } else {
                    confirmedDiarSegments.append(DiarizedSegment(
                        speaker: currentSpeaker, text: remaining
                    ))
                }
            }
        }
        diarizedSegments = confirmedDiarSegments
    }

    // MARK: - Audio Resampling

    private func resampleAudio(_ audio: MLXArray, from sourceSR: Int, to targetSR: Int) throws -> MLXArray {
        let samples = audio.asArray(Float.self)

        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sourceSR), channels: 1, interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(targetSR), channels: 1, interleaved: false
        ) else {
            throw NSError(domain: "STT", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio formats"])
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw NSError(domain: "STT", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
        }

        let inputFrameCount = AVAudioFrameCount(samples.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
            throw NSError(domain: "STT", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create input buffer"])
        }
        inputBuffer.frameLength = inputFrameCount
        memcpy(inputBuffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        let ratio = Double(targetSR) / Double(sourceSR)
        let outputFrameCount = AVAudioFrameCount(Double(samples.count) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "STT", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create output buffer"])
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let error { throw error }

        let outputSamples = Array(UnsafeBufferPointer(
            start: outputBuffer.floatChannelData![0], count: Int(outputBuffer.frameLength)
        ))
        return MLXArray(outputSamples)
    }
}
