//
//  File.swift
//  MetalSplatter
//
//  Created by Trupthi P on 06/11/25.
//

import Foundation

public class SPZSceneReader {
    public static func read(from url: URL) async throws -> [SplatScenePoint] {
        print("🔄 Reading SPZ file: \(url.lastPathComponent)")
        
        // Parse the SPZ file
        let parseResult = try SPZParser.parse(fileURL: url)
        
        print("✅ Parsed \(parseResult.splats.count) splats from SPZ")
        
        // Create format capabilities based on the parsed header
        let capabilities = StandardFormatCapabilities.spz(degree: Int(parseResult.header.shDegree))
        
        // Convert to SplatScenePoint array with capabilities
        let splatScenePoints = parseResult.splats.map { $0.toSplatScenePoint(with: capabilities) }
        
        return splatScenePoints
    }
}
