#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant const int kMaxViewCount = 2;
constant static const half kBoundsRadius = 3;
constant static const half kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

enum BufferIndex: int32_t
{
    BufferIndexUniforms = 0,
    BufferIndexSplat    = 1,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    uint2 screenSize;

    /*
     The first N splats are represented as as 2N primitives and 4N vertex indices. The remained are represented
     as instanced of these first N. This allows us to limit the size of the indexed array (and associated memory),
     but also avoid the performance penalty of a very large number of instances.
     */
    uint splatCount;
    uint indexedSplatCount;
    
    // Spherical harmonics configuration
    uint useSphericalHarmonics;  // 1 = enabled, 0 = disabled (fallback to basic color)
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

typedef struct
{
    packed_float3 position;
    packed_half4 color;        // SH[0] coefficients (RGB + opacity)
    packed_half3 covA;
    packed_half3 covB;
    // Additional spherical harmonics coefficients (SH[1-15]) stored as half precision
    packed_half3 sh1;          // SH[1] RGB
    packed_half3 sh2;          // SH[2] RGB  
    packed_half3 sh3;          // SH[3] RGB
    packed_half3 sh4;          // SH[4] RGB
    packed_half3 sh5;          // SH[5] RGB
    packed_half3 sh6;          // SH[6] RGB
    packed_half3 sh7;          // SH[7] RGB
    packed_half3 sh8;          // SH[8] RGB
    packed_half3 sh9;          // SH[9] RGB
    packed_half3 sh10;         // SH[10] RGB
    packed_half3 sh11;         // SH[11] RGB
    packed_half3 sh12;         // SH[12] RGB
    packed_half3 sh13;         // SH[13] RGB
    packed_half3 sh14;         // SH[14] RGB
    packed_half3 sh15;         // SH[15] RGB
} Splat;

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
} FragmentIn;
