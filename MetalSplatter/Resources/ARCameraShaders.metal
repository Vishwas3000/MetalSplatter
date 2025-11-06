#include <metal_stdlib>
using namespace metal;

struct ARCameraVertexIn {
    float2 position;
    float2 texCoord;
};

struct CameraTransform {
    float3x3 displayTransform;
};

struct ARCameraVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex ARCameraVertexOut arCameraVertexShader(const device ARCameraVertexIn* vertices [[buffer(0)]],
                                               const device CameraTransform& transform [[buffer(1)]],
                                               uint vid [[vertex_id]]) {
    ARCameraVertexOut out;
    ARCameraVertexIn vert = vertices[vid];
    
    // Position camera background at far depth so splats render in front
    out.position = float4(vert.position, 1.0, 1.0);  // z = 1.0, w = 1.0 (far plane)
    
    // Apply display transform to handle device orientation
    float3 transformedTexCoord = transform.displayTransform * float3(vert.texCoord, 1.0);
    out.texCoord = transformedTexCoord.xy;
    
    return out;
}

// Apple's full-range YCbCr to RGB conversion matrix (ITU-T T.871 specification)
constant float4x4 ycbcrToRGBTransform = float4x4(
    float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
    float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f), 
    float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
    float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f)
);

// sRGB to linear conversion function
float srgbToLinear(float color) {
    if (color <= 0.04045) {
        return color / 12.92;
    } else {
        return pow((color + 0.055) / 1.055, 2.4);
    }
}

// Linear to sRGB conversion function  
float linearToSrgb(float color) {
    if (color <= 0.0031308) {
        return color * 12.92;
    } else {
        return 1.055 * pow(color, 1.0/2.4) - 0.055;
    }
}

float3 srgbToLinear(float3 color) {
    return float3(srgbToLinear(color.r), srgbToLinear(color.g), srgbToLinear(color.b));
}

float3 linearToSrgb(float3 color) {
    return float3(linearToSrgb(color.r), linearToSrgb(color.g), linearToSrgb(color.b));
}

fragment float4 arCameraFragmentShader(ARCameraVertexOut in [[stage_in]],
                                       texture2d<float> yTexture [[texture(0)]],
                                       texture2d<float> uvTexture [[texture(1)]]) {
    constexpr sampler textureSampler(mag_filter::linear,
                                     min_filter::linear,
                                     address::clamp_to_edge);
    
    // Use texture coordinates for aspect-fill (crop-to-fill) behavior
    float2 texCoord = in.texCoord;
    
    // Check if we have YUV textures by checking if UV texture is bound and has valid dimensions
    if (uvTexture.get_width() > 0 && uvTexture.get_height() > 0) {
        // Sample YUV components using transformed coordinates
        float y = yTexture.sample(textureSampler, texCoord).r;
        float2 uv = uvTexture.sample(textureSampler, texCoord).rg;
        
        // Convert YUV to RGB using Apple's official matrix
        float4 ycbcr = float4(y, uv.r, uv.g, 1.0);
        float4 rgbResult = ycbcrToRGBTransform * ycbcr;
        
        // Clamp RGB values to valid range
        float3 rgb = clamp(rgbResult.rgb, 0.0, 1.0);
        
        // Convert from sRGB to linear for proper color space handling
        rgb = srgbToLinear(rgb);
        
        return float4(rgb, 1.0);
    } else {
        // Direct RGB/BGRA texture (fallback)
        float4 color = yTexture.sample(textureSampler, texCoord);
        
        // Convert to linear color space
        float3 linearColor = srgbToLinear(color.rgb);
        
        return float4(linearColor, 1.0);
    }
}

fragment float4 arCameraFragmentShaderRGB(ARCameraVertexOut in [[stage_in]],
                                          texture2d<float> rgbTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear,
                                     min_filter::linear,
                                     address::clamp_to_edge);
    
    float4 color = rgbTexture.sample(textureSampler, in.texCoord);
    
    // Convert from sRGB to linear color space for consistent rendering
    float3 linearColor = srgbToLinear(color.rgb);
    
    return float4(linearColor, 1.0);  // Ensure alpha = 1.0 for opaque camera background
}
