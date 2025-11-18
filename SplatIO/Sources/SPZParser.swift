//
//  SpzPraser.swift
//  GaussianSplat
//
//  Created by Trupthi P on 08/10/25.
//

import Foundation
import simd
import zlib
import SplatIO


public class SPZParser {
    
    public enum SPZError: Error {
        case invalidFile
        case decompressionFailed
        case invalidHeader
        case unsupportedVersion
        case dataSizeMismatch
    }
    
    // MARK: - Header Structure
    public struct GaussiansHeader {
        let magic: UInt32
        let version: UInt32
        let numPoints: UInt32
        let shDegree: UInt8
        let fractionalBits: UInt8
        let flags: UInt8
        let reserved: UInt8
        
        static func parse(from data: Data) -> GaussiansHeader? {
            guard data.count >= 16 else { return nil }
            
            var offset = 0
            let magic = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            
            let version = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            
            let numPoints = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            
            let shDegree = data[offset]
            offset += 1
            
            let fractionalBits = data[offset]
            offset += 1
            
            let flags = data[offset]
            offset += 1
            
            let reserved = data[offset]
            
            return GaussiansHeader(
                magic: magic,
                version: version,
                numPoints: numPoints,
                shDegree: shDegree,
                fractionalBits: fractionalBits,
                flags: flags,
                reserved: reserved
            )
        }
        
        func validate() -> Bool {
            guard magic == 0x5053474E else {
                print(" Invalid magic number")
                return false
            }
            
            guard version == 2 || version == 3 else {
                print("Unsupported version: \(version)")
                return false
            }
            
            guard shDegree <= 3 else {
                print(" Invalid SH degree: \(shDegree)")
                return false
            }
            
            return true
        }
    }
    
    public struct SplatData {
        var position: SIMD3<Float>
        
        var rotation: simd_quatf
        
        var scale: SIMD3<Float>
        
        var color: SIMD3<Float>
        var opacity: Float
        
        var sphericalHarmonics: [SIMD3<Float>] = [] // SH coefficients beyond degree 0
        
        var depth: Float = 0
    }
    
    public struct BoundingBox {
        var min: SIMD3<Float>
        var max: SIMD3<Float>
    }
    
    public struct ParseResult {
        public let splats: [SplatData]
        public let bounds: BoundingBox
        public let header: GaussiansHeader
    }
    
    
    public static func parse(fileURL: URL) throws -> ParseResult {
        print(" SPZ PARSER - Direct Storage (No Packing)")
        
        let compressedData = try Data(contentsOf: fileURL)
        print(" Loaded compressed file: \(compressedData.count) bytes")
        
        // Verify GZIP signature
        guard compressedData.prefix(2).starts(with: [0x1F, 0x8B]) else {
            throw SPZError.invalidFile
        }
        
        // Decompress
        let decompressed = try decompressGZIP(compressedData)
        print("✓ Decompressed: \(decompressed.count) bytes")
        
        // Parse
        let result = try parseDecompressedData(decompressed)
        
        print("Successfully parsed \(result.splats.count) splats")
        
        return result
    }
    
    // MARK: - GZIP Decompression
    
    private static func decompressGZIP(_ data: Data) throws -> Data {
        let estimatedSize = data.count * 15
        var outputBuffer = Data(count: estimatedSize)
        var actualSize: Int = 0
        
        let result = data.withUnsafeBytes { srcPtr -> Int32 in
            outputBuffer.withUnsafeMutableBytes { destPtr -> Int32 in
                var stream = z_stream()
                stream.next_in = UnsafeMutablePointer(mutating: srcPtr.baseAddress?.assumingMemoryBound(to: UInt8.self))
                stream.avail_in = UInt32(data.count)
                stream.next_out = destPtr.baseAddress?.assumingMemoryBound(to: UInt8.self)
                stream.avail_out = UInt32(estimatedSize)
                
                var status = inflateInit2_(&stream, 15 + 16, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
                guard status == Z_OK else { return -1 }
                
                status = inflate(&stream, Z_FINISH)
                actualSize = Int(stream.total_out)
                inflateEnd(&stream)
                
                return status == Z_STREAM_END ? Int32(actualSize) : -1
            }
        }
        
        guard result > 0 else {
            throw SPZError.decompressionFailed
        }
        
        outputBuffer.count = actualSize
        return outputBuffer
    }
    
    // MARK: - Parse Decompressed Data
    
    private static func parseDecompressedData(_ data: Data) throws -> ParseResult {
        // Parse header
        guard let header = GaussiansHeader.parse(from: data) else {
            throw SPZError.invalidHeader
        }
        
        guard header.validate() else {
            throw SPZError.invalidHeader
        }
        
        print("\n📋 Header Info:")
        print("   Version: \(header.version)")
        print("   Points: \(header.numPoints)")
        print("   SH Degree: \(header.shDegree)")
        print("   Fractional Bits: \(header.fractionalBits)")
        
        let numPoints = Int(header.numPoints)
        var offset = 16
        
        // Calculate expected data sections
        let positionsSize = numPoints * 9
        let alphasSize = numPoints * 1
        let colorsSize = numPoints * 3
        let scalesSize = numPoints * 3
        let rotationsSize = numPoints * (header.version == 3 ? 4 : 3)
        
        // Calculate spherical harmonics size
        // For degree n: total coefficients = (n+1)^2, but colors already contain SH[0]
        // So additional SH coefficients = (n+1)^2 - 1
        let shDegree = Int(header.shDegree)
        let additionalSHCoefficients = shDegree > 0 ? ((shDegree + 1) * (shDegree + 1)) - 1 : 0
        let shSize = numPoints * additionalSHCoefficients * 3 // 3 color channels (RGB)
        
        let expectedSize = 16 + positionsSize + alphasSize + colorsSize + scalesSize + rotationsSize + shSize
        
        print("\n📊 Data Layout:")
        print("   Positions: \(positionsSize) bytes")
        print("   Alphas: \(alphasSize) bytes")
        print("   Colors: \(colorsSize) bytes")
        print("   Scales: \(scalesSize) bytes")
        print("   Rotations: \(rotationsSize) bytes")
        print("   SH Coefficients (degree \(shDegree)): \(shSize) bytes (\(additionalSHCoefficients) coeffs/point)")
        print("   Expected: \(expectedSize) bytes, Actual: \(data.count) bytes")
        
        guard data.count >= expectedSize else {
            throw SPZError.dataSizeMismatch
        }
        
        // Extract data sections
        let positionsData = data.subdata(in: offset..<(offset + positionsSize))
        offset += positionsSize
        
        let alphasData = data.subdata(in: offset..<(offset + alphasSize))
        offset += alphasSize
        
        let colorsData = data.subdata(in: offset..<(offset + colorsSize))
        offset += colorsSize
        
        let scalesData = data.subdata(in: offset..<(offset + scalesSize))
        offset += scalesSize
        
        let rotationsData = data.subdata(in: offset..<(offset + rotationsSize))
        offset += rotationsSize
        
        // Extract spherical harmonics data if present
        let shData: Data?
        if shSize > 0 {
            shData = data.subdata(in: offset..<(offset + shSize))
            offset += shSize
        } else {
            shData = nil
        }
        
        print("\n⚙️  Processing splats...")
        
        // Calculate bounding box
        let bounds = calculateBounds(
            positions: positionsData,
            numPoints: numPoints,
            fractionalBits: header.fractionalBits
        )
        
        print("   Bounds: min(\(String(format: "%.2f", bounds.min.x)), \(String(format: "%.2f", bounds.min.y)), \(String(format: "%.2f", bounds.min.z)))")
        print("           max(\(String(format: "%.2f", bounds.max.x)), \(String(format: "%.2f", bounds.max.y)), \(String(format: "%.2f", bounds.max.z)))")
        
        // Convert to simple splats (no packing)
        let splats = try convertToSimpleSplats(
            positions: positionsData,
            alphas: alphasData,
            colors: colorsData,
            scales: scalesData,
            rotations: rotationsData,
            sphericalHarmonics: shData,
            numPoints: numPoints,
            header: header
        )
        
        return ParseResult(splats: splats, bounds: bounds, header: header)
    }
    
    // MARK: - Calculate Bounding Box
    
    private static func calculateBounds(
        positions: Data,
        numPoints: Int,
        fractionalBits: UInt8
    ) -> BoundingBox {
        let scaleFactor = Float(1 << fractionalBits)
        
        var minPos = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxPos = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
        
        for i in 0..<numPoints {
            let offset = i * 9
            
            // Read 24-bit positions
            let rawX = UInt32(positions[offset]) | (UInt32(positions[offset + 1]) << 8) | (UInt32(positions[offset + 2]) << 16)
            let rawY = UInt32(positions[offset + 3]) | (UInt32(positions[offset + 4]) << 8) | (UInt32(positions[offset + 5]) << 16)
            let rawZ = UInt32(positions[offset + 6]) | (UInt32(positions[offset + 7]) << 8) | (UInt32(positions[offset + 8]) << 16)
            
            // Sign extend and convert to float
            let signedX = Int32(bitPattern: rawX << 8) >> 8
            let signedY = Int32(bitPattern: rawY << 8) >> 8
            let signedZ = Int32(bitPattern: rawZ << 8) >> 8
            
            let x = Float(signedX) / scaleFactor
            let y = Float(signedY) / scaleFactor
            let z = Float(signedZ) / scaleFactor
            
            minPos = min(minPos, SIMD3<Float>(x, y, z))
            maxPos = max(maxPos, SIMD3<Float>(x, y, z))
        }
        
        // Apply Y-flip
        let temp = minPos.y
        minPos.y = -maxPos.y
        maxPos.y = -temp
        
        return BoundingBox(min: minPos, max: maxPos)
    }
    
    // MARK: - Convert to Simple Splats (No Packing)
    
    private static func convertToSimpleSplats(
        positions: Data,
        alphas: Data,
        colors: Data,
        scales: Data,
        rotations: Data,
        sphericalHarmonics: Data?,
        numPoints: Int,
        header: GaussiansHeader
    ) throws -> [SplatData] {
        var splats = [SplatData]()
        splats.reserveCapacity(numPoints)
        
        let scaleFactor = Float(1 << header.fractionalBits)
        
        for i in 0..<numPoints {
            // === POSITION (9 bytes - decode to world space floats) ===
            let posOffset = i * 9
            let rawX = UInt32(positions[posOffset]) | (UInt32(positions[posOffset + 1]) << 8) | (UInt32(positions[posOffset + 2]) << 16)
            let rawY = UInt32(positions[posOffset + 3]) | (UInt32(positions[posOffset + 4]) << 8) | (UInt32(positions[posOffset + 5]) << 16)
            let rawZ = UInt32(positions[posOffset + 6]) | (UInt32(positions[posOffset + 7]) << 8) | (UInt32(positions[posOffset + 8]) << 16)
            
            let signedX = Int32(bitPattern: rawX << 8) >> 8
            let signedY = Int32(bitPattern: rawY << 8) >> 8
            let signedZ = Int32(bitPattern: rawZ << 8) >> 8
            
            let x = Float(signedX) / scaleFactor
            let y = Float(signedY) / scaleFactor  // Y-flip
            let z = Float(signedZ) / scaleFactor
            
            let position = SIMD3<Float>(x, y, z)
            
            // === ROTATION (3 or 4 bytes - decode to quaternion) ===
            let rotOffset = i * (header.version == 3 ? 4 : 3)
            let rotation = parseRotation(data: rotations, offset: rotOffset, version: header.version)
            
            // === SCALE (3 bytes - decode from log space to linear) ===
            let scaleOffset = i * 3
            let scaleX = exp(Float(scales[scaleOffset]) / 255.0 * 10.0 - 5.0)
            let scaleY = exp(Float(scales[scaleOffset + 1]) / 255.0 * 10.0 - 5.0)
            let scaleZ = exp(Float(scales[scaleOffset + 2]) / 255.0 * 10.0 - 5.0)
            
            let scale = SIMD3<Float>(
                clamp(scaleX, 0.001, 10.0),
                clamp(scaleY, 0.001, 10.0),
                clamp(scaleZ, 0.001, 10.0)
            )
            
            // === COLOR & OPACITY (4 bytes - convert to 0-1 range) ===
            let colorOffset = i * 3
            let color = SIMD3<Float>(
                Float(colors[colorOffset]) / 255.0,
                Float(colors[colorOffset + 1]) / 255.0,
                Float(colors[colorOffset + 2]) / 255.0
            )
            
            let opacity = Float(alphas[i]) / 255.0
            
            // === SPHERICAL HARMONICS (parse additional SH coefficients beyond degree 0) ===
            var shCoefficients: [SIMD3<Float>] = []
            
            if let shData = sphericalHarmonics, header.shDegree > 0 {
                let additionalSHCoefficients = ((Int(header.shDegree) + 1) * (Int(header.shDegree) + 1)) - 1
                
                if i == 0 {
                    print("🌈 SPZ: Using Spherical Harmonics with \(additionalSHCoefficients + 1) coefficients (degree \(header.shDegree))")
                    
                    // Log first few coefficient values to check ranges
                    let baseIdx = 0
                    let rawR = Int8(bitPattern: shData[baseIdx])
                    let rawG = Int8(bitPattern: shData[baseIdx + 1]) 
                    let rawB = Int8(bitPattern: shData[baseIdx + 2])
                    
                    let firstCoeffR = Float(rawR) / 127.0
                    let firstCoeffG = Float(rawG) / 127.0
                    let firstCoeffB = Float(rawB) / 127.0
                    print("📊 First SH coefficient (SH[1]) RGB: (\(String(format: "%.3f", firstCoeffR)), \(String(format: "%.3f", firstCoeffG)), \(String(format: "%.3f", firstCoeffB)))")
                    print("📊 Raw SH bytes: (\(rawR), \(rawG), \(rawB))")
                }
                
                // Each point has additionalSHCoefficients * 3 bytes (RGB for each coefficient)
                let shOffset = i * additionalSHCoefficients * 3
                
                // Parse SH coefficients (8-bit signed integers)
                for coeffIdx in 0..<additionalSHCoefficients {
                    let baseIdx = shOffset + coeffIdx * 3
                    
                    // Convert 8-bit signed integers to floats (proper SPZ encoding)
                    // SPZ uses signed 8-bit integers in range [-127, +127] mapped to [-1.0, +1.0]
                    let r = Float(Int8(bitPattern: shData[baseIdx])) / 127.0
                    let g = Float(Int8(bitPattern: shData[baseIdx + 1])) / 127.0
                    let b = Float(Int8(bitPattern: shData[baseIdx + 2])) / 127.0
                    
                    shCoefficients.append(SIMD3<Float>(r, g, b))
                }
            } else if i == 0 {
                print("📦 SPZ: Using basic color only (no spherical harmonics)")
            }
            
            // 🔍 DEBUG: Log first 3 splats for validation (reduced logging)
            if i < 3 {
                print("\n🔍 SPZ Splat \(i) Raw Data:")
                print("   Position: (\(String(format: "%.4f", x)), \(String(format: "%.4f", y)), \(String(format: "%.4f", z)))")
                print("   Base Color (SH[0]): (\(String(format: "%.3f", color.x)), \(String(format: "%.3f", color.y)), \(String(format: "%.3f", color.z)))")
                print("   Opacity: \(String(format: "%.3f", opacity))")
                print("   Scale: (\(String(format: "%.4f", scale.x)), \(String(format: "%.4f", scale.y)), \(String(format: "%.4f", scale.z)))")
                if !shCoefficients.isEmpty {
                    print("   Additional SH Coefficients (\(shCoefficients.count)):")
                    for (idx, coeff) in shCoefficients.prefix(5).enumerated() {
                        print("     SH[\(idx+1)]: (\(String(format: "%.3f", coeff.x)), \(String(format: "%.3f", coeff.y)), \(String(format: "%.3f", coeff.z)))")
                    }
                }
            }
            
            // Create simple splat (no packing!)
            let splat = SplatData(
                position: position,
                rotation: rotation,
                scale: scale,
                color: color,
                opacity: opacity,
                sphericalHarmonics: shCoefficients,
                depth: 0
            )
            
            splats.append(splat)  // ✅ Append parsed splat
        }
        
        // 📊 VALIDATION: Analyze parsed data ranges for quality check
        print("\n📊 SPZ Data Validation Summary:")
        
        var positionBounds = (min: SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude),
                             max: SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude))
        var colorBounds = (min: SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude),
                          max: SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude))
        var scaleBounds = (min: SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude),
                          max: SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude))
        var opacityBounds = (min: Float.greatestFiniteMagnitude, max: -Float.greatestFiniteMagnitude)
        
        var invalidCount = 0
        var zeroOpacityCount = 0
        var blackSplatCount = 0
        
        for splat in splats {
            // Track bounds
            positionBounds.min = min(positionBounds.min, splat.position)
            positionBounds.max = max(positionBounds.max, splat.position)
            colorBounds.min = min(colorBounds.min, splat.color)
            colorBounds.max = max(colorBounds.max, splat.color)
            scaleBounds.min = min(scaleBounds.min, splat.scale)
            scaleBounds.max = max(scaleBounds.max, splat.scale)
            opacityBounds.min = min(opacityBounds.min, splat.opacity)
            opacityBounds.max = max(opacityBounds.max, splat.opacity)
            
            // Count issues
            if splat.opacity <= 0.01 { zeroOpacityCount += 1 }
            if length(splat.color) < 0.1 { blackSplatCount += 1 }
            if splat.position.x.isNaN || splat.position.y.isNaN || splat.position.z.isNaN ||
               splat.color.x.isNaN || splat.color.y.isNaN || splat.color.z.isNaN ||
               splat.scale.x.isNaN || splat.scale.y.isNaN || splat.scale.z.isNaN ||
               splat.opacity.isNaN {
                invalidCount += 1
            }
        }
        
        print("   Position Range: (\(String(format: "%.3f", positionBounds.min.x)), \(String(format: "%.3f", positionBounds.min.y)), \(String(format: "%.3f", positionBounds.min.z))) to (\(String(format: "%.3f", positionBounds.max.x)), \(String(format: "%.3f", positionBounds.max.y)), \(String(format: "%.3f", positionBounds.max.z)))")
        print("   Color Range: (\(String(format: "%.3f", colorBounds.min.x)), \(String(format: "%.3f", colorBounds.min.y)), \(String(format: "%.3f", colorBounds.min.z))) to (\(String(format: "%.3f", colorBounds.max.x)), \(String(format: "%.3f", colorBounds.max.y)), \(String(format: "%.3f", colorBounds.max.z)))")
        print("   Scale Range: (\(String(format: "%.4f", scaleBounds.min.x)), \(String(format: "%.4f", scaleBounds.min.y)), \(String(format: "%.4f", scaleBounds.min.z))) to (\(String(format: "%.4f", scaleBounds.max.x)), \(String(format: "%.4f", scaleBounds.max.y)), \(String(format: "%.4f", scaleBounds.max.z)))")
        print("   Opacity Range: \(String(format: "%.3f", opacityBounds.min)) to \(String(format: "%.3f", opacityBounds.max))")
        print("   Quality Issues:")
        print("     Invalid (NaN) splats: \(invalidCount)")
        print("     Near-zero opacity splats: \(zeroOpacityCount) (\(String(format: "%.1f", Float(zeroOpacityCount) / Float(numPoints) * 100))%)")
        print("     Near-black splats: \(blackSplatCount) (\(String(format: "%.1f", Float(blackSplatCount) / Float(numPoints) * 100))%)")
       
        
        return splats
    }
    
    // MARK: - Rotation Parsing
    

    private static func parseRotation(data: Data, offset: Int, version: UInt32) -> simd_quatf {
        if version == 3 {
            return parseRotationV3(data: data, offset: offset)
        } else {
            return parseRotationV2(data: data, offset: offset)
        }
    }

    // MARK: - V3: 4-Byte (unchanged)

    private static func parseRotationV3(data: Data, offset: Int) -> simd_quatf {
        guard offset + 3 < data.count else {
            return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        
        let rotData = UInt32(data[offset]) |
                     (UInt32(data[offset + 1]) << 8) |
                     (UInt32(data[offset + 2]) << 16) |
                     (UInt32(data[offset + 3]) << 24)
        
        let largestIdx = Int(rotData & 0x3)
        let compData = rotData >> 2
        
        var components: [Float] = [0, 0, 0, 0]
        var shift = 0
        
        for j in 0..<4 {
            if j != largestIdx {
                let val = Int16((compData >> shift) & 0x3FF) - 512
                components[j] = Float(val) / 512.0
                shift += 10
            }
        }
        
        let sumSq = components[0] * components[0] +
                    components[1] * components[1] +
                    components[2] * components[2] +
                    components[3] * components[3]
        components[largestIdx] = sqrt(max(0.0, 1.0 - sumSq))
        
        if components[largestIdx] < 0 {
            components[largestIdx] = -components[largestIdx]
        }
        
        return simd_quatf(ix: components[1], iy: components[2], iz: components[3], r: components[0]).normalized
    }

    // MARK: - V2: Try treating bytes as UNSIGNED (0-255) centered at 128

    private static func parseRotationV2(data: Data, offset: Int) -> simd_quatf {
        guard offset + 2 < data.count else {
            return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        
        let byte0 = data[offset]
        let byte1 = data[offset + 1]
        let byte2 = data[offset + 2]
        
        // This maps: 0→-1, 128→0, 255→1
        let x = (Float(byte0) - 128.0) / 128.0
        let y = (Float(byte1) - 128.0) / 128.0
        let z = (Float(byte2) - 128.0) / 128.0
        
        // Reconstruct w from unit quaternion constraint
        let sumSq = x * x + y * y + z * z
        

        if sumSq <= 1.0 {
            let w = sqrt(1.0 - sumSq)
            return simd_quatf(ix: x, iy: y, iz: z, r: w).normalized
        } else {
            // If outside unit sphere, normalize and reconstruct
            let normFactor = sqrt(0.999 / sumSq)
            let nx = x * normFactor
            let ny = y * normFactor
            let nz = z * normFactor
            let w = sqrt(max(0.0, 1.0 - (nx*nx + ny*ny + nz*nz)))
            return simd_quatf(ix: nx, iy: ny, iz: nz, r: w).normalized
        }
    }
    private static func clamp(_ value: Float, _ min: Float, _ max: Float) -> Float {
        return Swift.max(min, Swift.min(max, value))
    }
}

// MARK: - Extension for SplatScenePoint Conversion

extension SPZParser.SplatData {
    private static var conversionCount = 0
    
    public func toSplatScenePoint() -> SplatScenePoint {
        // Create proper spherical harmonics color representation
        let splatColor: SplatScenePoint.Color
        
        if sphericalHarmonics.isEmpty {
            // No additional SH coefficients - use basic linear color (SH degree 0 only)
            splatColor = .linearFloat(self.color)
        } else {
            // Combine SH[0] (base color) with additional SH coefficients
            var allSHCoefficients = [self.color]  // SH[0] from color field
            allSHCoefficients.append(contentsOf: sphericalHarmonics)  // SH[1-15] from additional data
            splatColor = .sphericalHarmonic(allSHCoefficients)
            
            // Print random 100 samples of SH values for analysis
            Self.conversionCount += 1
            if Self.conversionCount <= 100 {
                print("🔬 SH Sample #\(Self.conversionCount):")
                print("   Position: (\(String(format: "%.2f", self.position.x)), \(String(format: "%.2f", self.position.y)), \(String(format: "%.2f", self.position.z)))")
                print("   Base color: (\(String(format: "%.3f", self.color.x)), \(String(format: "%.3f", self.color.y)), \(String(format: "%.3f", self.color.z)))")
                
                // Print first few additional SH coefficients
                for (idx, coeff) in sphericalHarmonics.prefix(5).enumerated() {
                    print("   SH[\(idx+1)]: (\(String(format: "%.3f", coeff.x)), \(String(format: "%.3f", coeff.y)), \(String(format: "%.3f", coeff.z)))")
                }
                
                // Calculate value ranges for analysis
                let sh1Values = sphericalHarmonics.prefix(5).flatMap { [$0.x, $0.y, $0.z] }
                if !sh1Values.isEmpty {
                    let minVal = sh1Values.min() ?? 0
                    let maxVal = sh1Values.max() ?? 0
                    print("   SH[1-5] range: [\(String(format: "%.3f", minVal)), \(String(format: "%.3f", maxVal))]")
                }
                print("")
            }
        }
        
        return SplatScenePoint(
            position: self.position,
            color: splatColor,  // Now includes full SH coefficient array
            opacity: .linearFloat(self.opacity),  // SPZ opacity is already in 0-1 range
            scale: .linearFloat(self.scale),  // Use original anisotropic scale
            rotation: rotation,  // Use the rotation quaternion directly
            isSpz: true
        )
    }
}
