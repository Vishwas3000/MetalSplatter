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
    out.position = float4(vert.position, 0.0, 1.0);
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
        // YUV to RGB conversion (ITU-R BT.709)
        float y = yTexture.sample(textureSampler, in.texCoord).r;
        float2 uv = uvTexture.sample(textureSampler, in.texCoord).rg - 0.5;
        
        float3 rgb;
        rgb.r = y + 1.402 * uv.g;
        rgb.g = y - 0.344136 * uv.r - 0.714136 * uv.g;
        rgb.b = y + 1.772 * uv.r;
        
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