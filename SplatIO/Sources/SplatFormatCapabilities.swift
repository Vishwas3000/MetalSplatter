import Foundation

/// Protocol defining the capabilities of different splat file formats
public protocol SplatFormatCapabilities {
    /// Whether this format supports spherical harmonics data
    var supportsSphericalHarmonics: Bool { get }
    
    /// The color data format used by this file format
    var colorFormat: ColorDataFormat { get }
    
    /// Maximum spherical harmonics degree supported (0 = no SH support)
    var maxSHDegree: Int { get }
    
    /// Human-readable format identifier
    var formatName: String { get }
}

/// Enum defining different color data storage formats
public enum ColorDataFormat {
    case uint8RGB              // 8-bit RGB values (0-255) - used by .splat files
    case floatRGB              // Float RGB values (0.0-1.0) - basic .ply files
    case sphericalHarmonics    // Full spherical harmonics coefficients - SH-enabled files
}

/// Concrete implementation of format capabilities
public struct StandardFormatCapabilities: SplatFormatCapabilities {
    public let supportsSphericalHarmonics: Bool
    public let colorFormat: ColorDataFormat
    public let maxSHDegree: Int
    public let formatName: String
    
    public init(
        supportsSphericalHarmonics: Bool,
        colorFormat: ColorDataFormat,
        maxSHDegree: Int,
        formatName: String
    ) {
        self.supportsSphericalHarmonics = supportsSphericalHarmonics
        self.colorFormat = colorFormat
        self.maxSHDegree = maxSHDegree
        self.formatName = formatName
    }
}

/// Predefined capabilities for common file formats
public extension StandardFormatCapabilities {
    /// .splat file format capabilities - basic RGB only, no SH support
    static let dotSplat = StandardFormatCapabilities(
        supportsSphericalHarmonics: false,
        colorFormat: .uint8RGB,
        maxSHDegree: 0,
        formatName: ".splat"
    )
    
    /// Basic .ply file format capabilities - float RGB, no SH support
    static let plyBasic = StandardFormatCapabilities(
        supportsSphericalHarmonics: false,
        colorFormat: .floatRGB,
        maxSHDegree: 0,
        formatName: ".ply (basic)"
    )
    
    /// SH-enabled .ply file format capabilities
    static func plyWithSH(degree: Int) -> StandardFormatCapabilities {
        StandardFormatCapabilities(
            supportsSphericalHarmonics: degree > 0,
            colorFormat: .sphericalHarmonics,
            maxSHDegree: degree,
            formatName: ".ply (SH degree \(degree))"
        )
    }
    
    /// .spz file format capabilities
    static func spz(degree: Int) -> StandardFormatCapabilities {
        StandardFormatCapabilities(
            supportsSphericalHarmonics: degree > 0,
            colorFormat: .sphericalHarmonics,
            maxSHDegree: degree,
            formatName: ".spz (SH degree \(degree))"
        )
    }
}