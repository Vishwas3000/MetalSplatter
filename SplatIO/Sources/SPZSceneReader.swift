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
        
        // Convert to SplatScenePoint array
        let splatScenePoints = parseResult.splats.map { $0.toSplatScenePoint() }
        
        return splatScenePoints
    }
}
