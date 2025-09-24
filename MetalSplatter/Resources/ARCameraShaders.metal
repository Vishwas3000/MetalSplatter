#include <metal_stdlib>
using namespace metal;

struct ARCameraVertexIn {
    float2 position;
    float2 texCoord;
};

struct ARCameraVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex ARCameraVertexOut arCameraVertexShader(const device ARCameraVertexIn* vertices [[buffer(0)]],
                                               uint vid [[vertex_id]]) {
    ARCameraVertexOut out;
    ARCameraVertexIn vert = vertices[vid];
    
    // CRITICAL: Position the camera background at FAR DEPTH (z = 1.0)
    // This ensures all 3D content renders in front of the camera background
    out.position = float4(vert.position, 1.0, 1.0);  // z = 1.0 (far plane)
    out.texCoord = vert.texCoord;
    return out;
}

fragment float4 arCameraFragmentShader(ARCameraVertexOut in [[stage_in]],
                                       texture2d<float> yTexture [[texture(0)]],
                                       texture2d<float> uvTexture [[texture(1)]]) {
    constexpr sampler textureSampler(mag_filter::linear,
                                     min_filter::linear);
    
    // Check if we have YUV textures (texture 1 is bound) or RGB texture (only texture 0)
    if (uvTexture.get_width() > 0) {
        // Proper YUV to RGB conversion for ARKit camera feed (ITU-R BT.709)
        float y = yTexture.sample(textureSampler, in.texCoord).r;
        float2 uv = uvTexture.sample(textureSampler, in.texCoord).rg - float2(0.5, 0.5);
        
        // Use proper BT.709 conversion matrix
        float3 rgb;
        rgb.r = y + 1.28033 * uv.g;
        rgb.g = y - 0.21482 * uv.r - 0.38059 * uv.g;
        rgb.b = y + 2.12798 * uv.r;
        
        // Clamp to valid range
        rgb = saturate(rgb);
        
        return float4(rgb, 1.0);
    } else {
        // Direct RGB/BGRA texture
        return yTexture.sample(textureSampler, in.texCoord);
    }
}

fragment float4 arCameraFragmentShaderRGB(ARCameraVertexOut in [[stage_in]],
                                          texture2d<float> rgbTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear,
                                     min_filter::linear);
    
    return rgbTexture.sample(textureSampler, in.texCoord);
}