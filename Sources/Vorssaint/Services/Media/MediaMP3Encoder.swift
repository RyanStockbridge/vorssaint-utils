// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vorssaint

import Accelerate
import Foundation

/// Writes MPEG-1 Layer III audio, the format people mean by MP3.
///
/// macOS plays MP3 but ships no encoder for it, and Vorssaint takes no external
/// dependencies, so the format is written here. The structure follows the
/// Shine encoder: constant bitrate, long blocks only and no psychoacoustic
/// model, so each granule is quantized with one step size, the finest the
/// bitrate can pay for. Two choices make up for the missing model. The channels
/// of a granule share that step, so a quiet channel hands its bits to a busy
/// one, and content above what the bitrate can carry is dropped before it
/// competes for bits with what people actually hear. The standard's tables
/// live in `MediaMP3Tables`.
final class MediaMP3Encoder {
    enum ConfigurationError: Error, Equatable {
        case unsupportedSampleRate
        case unsupportedBitRate
        case unsupportedChannelCount
    }

    static let samplesPerFrame = 1152
    /// In the order of the header's sampling_frequency field.
    static let sampleRates = [44_100, 48_000, 32_000]
    /// Kilobits per second, in the order of the header's bitrate_index from 1.
    static let bitRates = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]

    let sampleRate: Int
    let channelCount: Int
    let bitRate: Int

    private static let granuleSize = 576
    private static let maximumQuantized: Int32 = 15 + 8191
    /// part2_3_length is a 12-bit field.
    private static let maximumGranuleBits = 4095
    private static let smallestStep = -210
    private static let largestStep = 45

    private let sampleRateIndex: Int
    private let bitRateIndex: Int
    private let bands: [Int]
    private let lowpassLine: Int
    private let sideInfoBits: Int
    private var analyses: [Analysis]
    private var pending: [[Float]]
    private var paddingRemainder = 0
    private var framesWritten = 0

    init(sampleRate: Int, channelCount: Int, bitRate: Int) throws {
        guard let sampleRateIndex = Self.sampleRates.firstIndex(of: sampleRate) else {
            throw ConfigurationError.unsupportedSampleRate
        }
        guard let bitRateIndex = Self.bitRates.firstIndex(of: bitRate) else {
            throw ConfigurationError.unsupportedBitRate
        }
        guard channelCount == 1 || channelCount == 2 else {
            throw ConfigurationError.unsupportedChannelCount
        }
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.bitRate = bitRate
        self.sampleRateIndex = sampleRateIndex
        self.bitRateIndex = bitRateIndex + 1
        bands = MediaMP3Tables.longBlockBands[sampleRateIndex]
        let perChannel = Double(bitRate) / Double(channelCount)
        let cutoff = min(20_000, max(8_000, 16_000 + (perChannel - 64) * 62.5))
        lowpassLine = min(Self.granuleSize,
                          Int((cutoff / (Double(sampleRate) / 2) * Double(Self.granuleSize)).rounded()))
        sideInfoBits = channelCount == 1 ? 136 : 256
        analyses = Array(repeating: Analysis(), count: channelCount)
        pending = Array(repeating: [], count: channelCount)
    }

    /// Takes interleaved samples in -1...1 and returns the frames they complete.
    func encode(interleaved samples: UnsafeBufferPointer<Float>) -> Data {
        let frames = samples.count / channelCount
        for channel in 0..<channelCount {
            pending[channel].reserveCapacity(pending[channel].count + frames)
            for frame in 0..<frames {
                pending[channel].append(samples[frame * channelCount + channel])
            }
        }
        return drain()
    }

    /// Ends the stream. The filterbank holds back about one frame of samples,
    /// so a frame of silence follows the last partial one to bring them out.
    /// Players recognize MP3 by two consecutive frames, so even an empty
    /// stream ends with two.
    func finish() -> Data {
        let remainder = pending[0].count % Self.samplesPerFrame
        var padding = (remainder == 0 ? 0 : Self.samplesPerFrame - remainder) + Self.samplesPerFrame
        if framesWritten + (pending[0].count + padding) / Self.samplesPerFrame < 2 {
            padding += Self.samplesPerFrame
        }
        for channel in 0..<channelCount {
            pending[channel].append(contentsOf: repeatElement(0, count: padding))
        }
        return drain()
    }

    private func drain() -> Data {
        var output = Data()
        var offset = 0
        while pending[0].count - offset >= Self.samplesPerFrame {
            output.append(contentsOf: encodeFrame(at: offset))
            offset += Self.samplesPerFrame
            framesWritten += 1
        }
        if offset > 0 {
            for channel in 0..<channelCount { pending[channel].removeFirst(offset) }
        }
        return output
    }

    // MARK: - Frame

    private func encodeFrame(at offset: Int) -> [UInt8] {
        let unit = 144_000 * bitRate
        paddingRemainder += unit % sampleRate
        let padding = paddingRemainder >= sampleRate ? 1 : 0
        paddingRemainder -= padding * sampleRate
        let frameBytes = unit / sampleRate + padding
        let mainBits = frameBytes * 8 - 32 - sideInfoBits

        var spectra = [[[Float]]](repeating: [[Float]](repeating: [], count: channelCount), count: 2)
        for channel in 0..<channelCount {
            pending[channel].withUnsafeBufferPointer { samples in
                for granule in 0..<2 {
                    var lines = analyses[channel].granule(samples.baseAddress! + offset + granule * Self.granuleSize)
                    for line in lowpassLine..<Self.granuleSize { lines[line] = 0 }
                    spectra[granule][channel] = lines
                }
            }
        }

        let midSide = channelCount == 2 && prefersMidSide(spectra)
        if midSide {
            let scale = Float(0.5.squareRoot())
            for granule in 0..<2 {
                let left = spectra[granule][0], right = spectra[granule][1]
                for line in 0..<Self.granuleSize {
                    spectra[granule][0][line] = (left[line] + right[line]) * scale
                    spectra[granule][1][line] = (left[line] - right[line]) * scale
                }
            }
        }

        var codings: [[Coding]] = []
        var quantized: [[[Int32]]] = []
        var remaining = mainBits
        for granule in 0..<2 {
            let budget = granule == 0 ? mainBits / 2 : remaining
            let (granuleCodings, granuleValues) = allocate(spectra[granule], budget: budget)
            remaining -= granuleCodings.reduce(0) { $0 + $1.bits }
            codings.append(granuleCodings)
            quantized.append(granuleValues)
        }

        var main: [[BitWriter]] = []
        for granule in 0..<2 {
            var row: [BitWriter] = []
            for channel in 0..<channelCount {
                var writer = BitWriter()
                writeMainData(quantized[granule][channel], signs: spectra[granule][channel],
                              coding: codings[granule][channel], to: &writer)
                codings[granule][channel].bits = writer.bitCount
                row.append(writer)
            }
            main.append(row)
        }

        var frame = BitWriter()
        writeHeader(padding: padding, midSide: midSide, to: &frame)
        writeSideInfo(codings, to: &frame)
        for granule in 0..<2 {
            for channel in 0..<channelCount { frame.append(main[granule][channel]) }
        }
        // Whatever the granules left unused is ancillary data, which decoders
        // skip because the next frame's main data begins at its own header.
        frame.pad(toBits: frameBytes * 8)
        return frame.bytes
    }

    /// Mid/side pays off when the channels are alike, because the side channel
    /// then carries little and costs few bits. Coding cost grows with the log
    /// of each band's energy, so the layout with the smaller summed log wins.
    private func prefersMidSide(_ spectra: [[[Float]]]) -> Bool {
        var energies = [[Double]](repeating: [Double](repeating: 0, count: 4), count: bands.count - 1)
        var peak = 0.0
        for granule in 0..<2 {
            let left = spectra[granule][0], right = spectra[granule][1]
            for band in 0..<(bands.count - 1) {
                for line in bands[band]..<bands[band + 1] {
                    let l = Double(left[line]), r = Double(right[line])
                    energies[band][0] += l * l
                    energies[band][1] += r * r
                    energies[band][2] += (l + r) * (l + r) / 2
                    energies[band][3] += (l - r) * (l - r) / 2
                }
                let width = Double(bands[band + 1] - bands[band])
                peak = max(peak, energies[band].max()! / width)
            }
        }
        guard peak > 0 else { return false }
        let floor = peak * 1e-7
        var separate = 0.0, joint = 0.0
        for band in 0..<(bands.count - 1) {
            let width = Double(bands[band + 1] - bands[band])
            func cost(_ energy: Double) -> Double { width * log2(energy / width + floor) }
            separate += cost(energies[band][0]) + cost(energies[band][1])
            joint += cost(energies[band][2]) + cost(energies[band][3])
        }
        return joint < separate
    }

    // MARK: - Rate control

    /// Finds the finest step at which every channel of the granule fits the
    /// shared budget, so quantization noise sits at one level in all of them.
    private func allocate(_ spectra: [[Float]], budget: Int) -> ([Coding], [[Int32]]) {
        let magnitudes = spectra.map { lines in lines.map { line -> Float in
            let value = abs(line)
            return (value * value.squareRoot()).squareRoot()
        } }
        let peaks = magnitudes.map { $0.max() ?? 0 }
        var values = [[Int32]](repeating: [Int32](repeating: 0, count: Self.granuleSize), count: spectra.count)
        let silent = [Coding](repeating: Coding(), count: spectra.count)

        func evaluate(_ step: Int) -> [Coding]? {
            let scale = Float(exp2(-0.1875 * Double(step)))
            var codings = silent
            var total = 0
            for channel in spectra.indices where peaks[channel] > 0 {
                guard peaks[channel] * scale + 0.4054 < Float(Self.maximumQuantized) + 1 else { return nil }
                magnitudes[channel].withUnsafeBufferPointer { source in
                    values[channel].withUnsafeMutableBufferPointer { target in
                        for line in 0..<Self.granuleSize {
                            target[line] = Int32(source[line] * scale + 0.4054)
                        }
                    }
                }
                var coding = code(values[channel])
                guard coding.bits <= Self.maximumGranuleBits else { return nil }
                coding.globalGain = step + 210
                codings[channel] = coding
                total += coding.bits
            }
            return total <= budget ? codings : nil
        }

        guard peaks.contains(where: { $0 > 0 }), evaluate(Self.largestStep) != nil else {
            return (silent, [[Int32]](repeating: [Int32](repeating: 0, count: Self.granuleSize),
                                      count: spectra.count))
        }
        var lower = Self.smallestStep, upper = Self.largestStep
        while lower < upper {
            let middle = lower + (upper - lower) / 2
            if evaluate(middle) != nil { upper = middle } else { lower = middle + 1 }
        }
        // The search can end on a step it only probed indirectly, so the
        // values and codings are rebuilt for the step actually chosen.
        let codings = evaluate(upper)!
        for channel in spectra.indices where peaks[channel] == 0 {
            values[channel] = [Int32](repeating: 0, count: Self.granuleSize)
        }
        return (codings, values)
    }

    /// Partitions quantized values into the big-value pairs, the count1
    /// quadruples and the zero tail, picks tables, and counts the bits.
    private func code(_ values: [Int32]) -> Coding {
        var coding = Coding()
        var end = Self.granuleSize
        while end > 1, values[end - 1] == 0, values[end - 2] == 0 { end -= 2 }
        var start = end
        while start > 3, values[start - 1] <= 1, values[start - 2] <= 1,
              values[start - 3] <= 1, values[start - 4] <= 1 {
            start -= 4
        }
        coding.bigValues = start / 2
        coding.count1 = (end - start) / 4

        var bitsA = 0, bitsB = 0
        for quad in 0..<coding.count1 {
            let base = start + quad * 4
            let index = Int(values[base] << 3 | values[base + 1] << 2 | values[base + 2] << 1 | values[base + 3])
            let signs = Int(values[base] + values[base + 1] + values[base + 2] + values[base + 3])
            bitsA += Int(Self.count1A.lengths[index]) + signs
            bitsB += 4 + signs
        }
        coding.count1Table = bitsB < bitsA ? 1 : 0

        let pairs = coding.bigValues * 2
        if pairs > 0 {
            var bandCount = 0
            while bands[bandCount] < pairs { bandCount += 1 }
            var region0 = Self.regionCounts[bandCount].0
            while region0 > 0, bands[region0 + 1] > pairs { region0 -= 1 }
            var region1 = Self.regionCounts[bandCount].1
            while region1 > 0, bands[region0 + region1 + 2] > pairs { region1 -= 1 }
            coding.region0Count = region0
            coding.region1Count = region1
        }
        let boundaries = regionBoundaries(coding)
        var bits = min(bitsA, bitsB)
        for region in 0..<3 {
            let choice = chooseTable(values, boundaries[region], boundaries[region + 1])
            coding.tables[region] = choice.table
            bits += choice.bits
        }
        coding.bits = bits
        return coding
    }

    /// Region starts as the decoder derives them, cut off where the big
    /// values end so a region past them is empty rather than miscounted.
    private func regionBoundaries(_ coding: Coding) -> [Int] {
        let pairs = coding.bigValues * 2
        let first = min(pairs, bands[coding.region0Count + 1])
        let second = min(pairs, bands[coding.region0Count + coding.region1Count + 2])
        return [0, first, second, pairs]
    }

    private func chooseTable(_ values: [Int32], _ begin: Int, _ end: Int) -> (table: Int, bits: Int) {
        guard begin < end else { return (0, 0) }
        var peak: Int32 = 0
        for index in begin..<end where values[index] > peak { peak = values[index] }
        guard peak > 0 else { return (0, 0) }
        let candidates: [Int]
        if peak < 15 {
            switch peak {
            case 1: candidates = [1]
            case 2: candidates = [2, 3]
            case 3: candidates = [5, 6]
            case 4, 5: candidates = [7, 8, 9]
            case 6, 7: candidates = [10, 11, 12]
            default: candidates = [13, 15]
            }
        } else {
            let escape = Int(peak) - 15
            let first = (16...23).first { (1 << Self.codebooks[$0]!.linbits) - 1 >= escape }!
            let second = (24...31).first { (1 << Self.codebooks[$0]!.linbits) - 1 >= escape }!
            candidates = [first, second]
        }
        var best = (table: candidates[0], bits: Int.max)
        for table in candidates {
            let bits = countBits(values, begin, end, table: table)
            if bits < best.bits { best = (table, bits) }
        }
        return best
    }

    private func countBits(_ values: [Int32], _ begin: Int, _ end: Int, table: Int) -> Int {
        let book = Self.codebooks[table]!
        var bits = 0
        var index = begin
        while index < end {
            var x = Int(values[index]), y = Int(values[index + 1])
            if book.linbits > 0 {
                if x > 14 { x = 15; bits += book.linbits }
                if y > 14 { y = 15; bits += book.linbits }
            }
            bits += Int(book.lengths[x * book.size + y])
            if x != 0 { bits += 1 }
            if y != 0 { bits += 1 }
            index += 2
        }
        return bits
    }

    // MARK: - Bitstream

    private func writeHeader(padding: Int, midSide: Bool, to writer: inout BitWriter) {
        writer.write(0x7FF, 11)          // sync
        writer.write(0b11, 2)            // MPEG-1
        writer.write(0b01, 2)            // Layer III
        writer.write(1, 1)               // no CRC
        writer.write(UInt32(bitRateIndex), 4)
        writer.write(UInt32(sampleRateIndex), 2)
        writer.write(UInt32(padding), 1)
        writer.write(0, 1)               // private
        writer.write(channelCount == 1 ? 0b11 : 0b01, 2)   // mono or joint stereo
        writer.write(midSide ? 0b10 : 0b00, 2)              // mid/side on, intensity off
        writer.write(0, 1)               // copyright
        writer.write(1, 1)               // original
        writer.write(0, 2)               // no emphasis
    }

    private func writeSideInfo(_ codings: [[Coding]], to writer: inout BitWriter) {
        writer.write(0, 9)                                // main_data_begin: no reservoir
        writer.write(0, channelCount == 1 ? 5 : 3)        // private bits
        writer.write(0, 4 * channelCount)                 // scfsi
        for granule in 0..<2 {
            for channel in 0..<channelCount {
                let coding = codings[granule][channel]
                writer.write(UInt32(coding.bits), 12)     // part2_3_length
                writer.write(UInt32(coding.bigValues), 9)
                writer.write(UInt32(coding.globalGain), 8)
                writer.write(0, 4)                        // scalefac_compress: no scalefactors
                writer.write(0, 1)                        // window_switching_flag: long blocks
                for region in 0..<3 { writer.write(UInt32(coding.tables[region]), 5) }
                writer.write(UInt32(coding.region0Count), 4)
                writer.write(UInt32(coding.region1Count), 3)
                writer.write(0, 1)                        // preflag
                writer.write(0, 1)                        // scalefac_scale
                writer.write(UInt32(coding.count1Table), 1)
            }
        }
    }

    private func writeMainData(_ values: [Int32], signs: [Float], coding: Coding, to writer: inout BitWriter) {
        let boundaries = regionBoundaries(coding)
        var index = 0
        while index < coding.bigValues * 2 {
            let region = index < boundaries[1] ? 0 : (index < boundaries[2] ? 1 : 2)
            let table = coding.tables[region]
            if table != 0 {
                writePair(Int(values[index]), signs[index] < 0, Int(values[index + 1]), signs[index + 1] < 0,
                          book: Self.codebooks[table]!, to: &writer)
            }
            index += 2
        }
        let book = coding.count1Table == 0 ? Self.count1A : Self.count1B
        for _ in 0..<coding.count1 {
            let quad = (0..<4).map { Int(values[index + $0]) }
            let code = quad[0] << 3 | quad[1] << 2 | quad[2] << 1 | quad[3]
            writer.write(UInt32(book.codes[code]), Int(book.lengths[code]))
            for offset in 0..<4 where quad[offset] != 0 {
                writer.write(signs[index + offset] < 0 ? 1 : 0, 1)
            }
            index += 4
        }
    }

    private func writePair(_ x: Int, _ xNegative: Bool, _ y: Int, _ yNegative: Bool,
                           book: MediaMP3Tables.Codebook, to writer: inout BitWriter) {
        let escapedX = book.linbits > 0 && x > 14
        let escapedY = book.linbits > 0 && y > 14
        let index = (escapedX ? 15 : x) * book.size + (escapedY ? 15 : y)
        writer.write(UInt32(book.codes[index]), Int(book.lengths[index]))
        if escapedX { writer.write(UInt32(x - 15), book.linbits) }
        if x != 0 { writer.write(xNegative ? 1 : 0, 1) }
        if escapedY { writer.write(UInt32(y - 15), book.linbits) }
        if y != 0 { writer.write(yNegative ? 1 : 0, 1) }
    }

    // MARK: - Tables

    private struct Coding {
        var bits = 0
        var bigValues = 0
        var count1 = 0
        var count1Table = 0
        var globalGain = 210
        var tables = [0, 0, 0]
        var region0Count = 0
        var region1Count = 0
    }

    private static let codebooks = MediaMP3Tables.codebooks
    private static let count1A = codebooks[32]!
    private static let count1B = codebooks[33]!
    private static let regionCounts = MediaMP3Tables.regionCounts
}

// MARK: - Analysis

private extension MediaMP3Encoder {
    /// One channel's polyphase filterbank and MDCT, carried across frames.
    struct Analysis {
        /// The last 512 samples, newest first, as the standard's X vector.
        private var history = [Float](repeating: 0, count: 512)
        /// The previous granule's subband samples, laid out [time * 32 + band].
        private var previous = [Float](repeating: 0, count: 576)
        private var windowed = [Float](repeating: 0, count: 512)
        private var folded = [Float](repeating: 0, count: 64)

        /// Turns 576 samples into 576 frequency lines, [band * 18 + line].
        mutating func granule(_ samples: UnsafePointer<Float>) -> [Float] {
            var current = [Float](repeating: 0, count: 576)
            for time in 0..<18 {
                history.withUnsafeMutableBufferPointer { buffer in
                    let base = buffer.baseAddress!
                    memmove(base + 32, base, 480 * MemoryLayout<Float>.stride)
                    for index in 0..<32 { base[31 - index] = samples[time * 32 + index] }
                }
                vDSP_vmul(history, 1, Filter.window, 1, &windowed, 1, 512)
                vDSP_mmul(Filter.ones, 1, windowed, 1, &folded, 1, 1, 64, 8)
                current.withUnsafeMutableBufferPointer { buffer in
                    let row = buffer.baseAddress! + time * 32
                    vDSP_mmul(Filter.polyphase, 1, folded, 1, row, 1, 32, 1, 64)
                    // The filterbank inverts the spectrum of every odd band;
                    // negating its odd samples puts the lines back in order.
                    if time % 2 == 1 {
                        for band in stride(from: 1, to: 32, by: 2) { row[band] = -row[band] }
                    }
                }
            }

            var input = [Float](repeating: 0, count: 32 * 36)
            for band in 0..<32 {
                for time in 0..<18 {
                    input[band * 36 + time] = previous[time * 32 + band]
                    input[band * 36 + 18 + time] = current[time * 32 + band]
                }
            }
            previous = current
            var lines = [Float](repeating: 0, count: 576)
            vDSP_mmul(input, 1, Filter.mdct, 1, &lines, 1, 32, 18, 36)

            // Undo the aliasing between neighbouring bands that the decoder
            // will reintroduce, butterfly by butterfly (Table B.9).
            for band in 1..<32 {
                for index in 0..<8 {
                    let upper = (band - 1) * 18 + 17 - index
                    let lower = band * 18 + index
                    let up = lines[upper], down = lines[lower]
                    lines[upper] = up * Filter.aliasCS[index] + down * Filter.aliasCA[index]
                    lines[lower] = down * Filter.aliasCS[index] - up * Filter.aliasCA[index]
                }
            }
            return lines
        }
    }

    enum Filter {
        static let window = MediaMP3Tables.analysisWindow
        static let ones = [Float](repeating: 1, count: 8)

        /// M[i][k] = cos((2i + 1)(16 - k) pi / 64), 32 rows of 64.
        static let polyphase: [Float] = {
            var matrix = [Float](repeating: 0, count: 32 * 64)
            for band in 0..<32 {
                for tap in 0..<64 {
                    let angle = Double((2 * band + 1) * (16 - tap)) * Double.pi / 64
                    matrix[band * 64 + tap] = Float(cos(angle))
                }
            }
            return matrix
        }()

        /// The sine-windowed long-block MDCT, transposed to 36 rows of 18 so one
        /// matrix product turns all 32 bands at once. Divided by 9, half the 18
        /// lines, because the decoder's inverse transform carries no scale.
        static let mdct: [Float] = {
            var matrix = [Float](repeating: 0, count: 36 * 18)
            for time in 0..<36 {
                let window = sin(Double.pi / 36 * (Double(time) + 0.5))
                for line in 0..<18 {
                    let angle = Double.pi / 72 * Double((2 * time + 19) * (2 * line + 1))
                    matrix[time * 18 + line] = Float(window * cos(angle) / 9)
                }
            }
            return matrix
        }()

        static let aliasCoefficients: [Double] = [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037]
        static let aliasCS = aliasCoefficients.map { Float(1 / (1 + $0 * $0).squareRoot()) }
        static let aliasCA = aliasCoefficients.map { Float($0 / (1 + $0 * $0).squareRoot()) }
    }

    /// Packs bits most significant first, as MPEG streams are written.
    struct BitWriter {
        private(set) var bytes: [UInt8] = []
        private var accumulator: UInt64 = 0
        private var pendingBits = 0

        var bitCount: Int { bytes.count * 8 + pendingBits }

        mutating func write(_ value: UInt32, _ count: Int) {
            guard count > 0 else { return }
            accumulator = accumulator << UInt64(count) | UInt64(value) & ((1 << UInt64(count)) - 1)
            pendingBits += count
            while pendingBits >= 8 {
                pendingBits -= 8
                bytes.append(UInt8(truncatingIfNeeded: accumulator >> UInt64(pendingBits)))
            }
            accumulator &= (1 << UInt64(pendingBits)) - 1
        }

        mutating func append(_ other: BitWriter) {
            for byte in other.bytes { write(UInt32(byte), 8) }
            write(UInt32(other.accumulator), other.pendingBits)
        }

        mutating func pad(toBits total: Int) {
            while bitCount < total { write(0, min(8, total - bitCount)) }
        }
    }
}
