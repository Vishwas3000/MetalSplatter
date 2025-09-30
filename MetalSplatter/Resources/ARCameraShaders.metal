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
    out.position = float4(vert.position, 1.0, 1.0);  // z = 1.0 (far plane)
    
    // Apply display transform to texture coordinates to handle device rotation
    float3 transformedTexCoord = transform.displayTransform * float3(vert.texCoord, 1.0);
    out.texCoord = transformedTexCoord.xy;
    
    return out;
}

fragment float4 arCameraFragmentShader(ARCameraVertexOut in [[stage_in]],
                                       texture2d<float> yTexture [[texture(0)]],
                                       texture2d<float> uvTexture [[texture(1)]]) {
    constexpr sampler textureSampler(mag_filter::linear,
                                     min_filter::linear,
                                     address::clamp_to_edge);
    
    // Check if we have YUV textures by checking if UV texture width > 0
    if (uvTexture.get_width() == 0) {
        // Direct RGB/BGRA texture
        float4 color = yTexture.sample(textureSampler, in.texCoord);
        return float4(color.rgb, 1.0);  // Ensure alpha = 1.0 for opaque camera background
    } else {
        // YUV to RGB conversion for ARKit camera feed (ITU-R BT.709 limited range)
        float y = yTexture.sample(textureSampler, in.texCoord).r;
        float2 uv = uvTexture.sample(textureSampler, in.texCoord).rg - float2(0.5, 0.5);
        
        // Correct BT.709 conversion matrix for limited range YUV
        float3 rgb;
        rgb.r = y + 1.5748 * uv.g;                    // Red component
        rgb.g = y - 0.1873 * uv.r - 0.4681 * uv.g;   // Green component  
        rgb.b = y + 1.8556 * uv.r;                    // Blue component
        
        // Clamp to valid range
        rgb = saturate(rgb);
        
        return float4(rgb, 1.0);  // Alpha = 1.0 for opaque camera background
    }
}

fragment float4 arCameraFragmentShaderRGB(ARCameraVertexOut in [[stage_in]],
                                          texture2d<float> rgbTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear,
                                     min_filter::linear);
    
    float4 color = rgbTexture.sample(textureSampler, in.texCoord);
    return float4(color.rgb, 1.0);  // Ensure alpha = 1.0 for opaque camera background
}