#import "SplatProcessing.h"

// Spherical harmonics coefficients for degrees 0-3 (16 total)
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

// Compute view-dependent color from spherical harmonics coefficients
half3 computeColorFromSH(Splat splat, float3 viewDirection) {
    // Degree 0 (constant term) - base SH coefficient with SH_C0 scaling
    half3 result = splat.color.rgb;
    
//    float x = viewDirection.x;
//    float y = viewDirection.y;
//    float z = viewDirection.z;
//    
//    // Degree 1 (linear terms) - 3 coefficients (SH[1-3])
//    result += half(SH_C1) * (-y * half3(splat.sh1) + z * half3(splat.sh2) - x * half3(splat.sh3));
//    
//    float xx = x * x, yy = y * y, zz = z * z;
//    float xy = x * y, yz = y * z, xz = x * z;
//    
//    // Degree 2 (quadratic terms) - 5 coefficients (SH[4-8])
//    result += half(SH_C2[0]) * xy * half3(splat.sh4);
//    result += half(SH_C2[1]) * yz * half3(splat.sh5);
//    result += half(SH_C2[2]) * (2.0f * zz - xx - yy) * half3(splat.sh6);
//    result += half(SH_C2[3]) * xz * half3(splat.sh7);
//    result += half(SH_C2[4]) * (xx - yy) * half3(splat.sh8);
//    
//    // Degree 3 (cubic terms) - 7 coefficients (SH[9-15])
//    result += half(SH_C3[0]) * y * (3.0f * xx - yy) * half3(splat.sh9);
//    result += half(SH_C3[1]) * xy * z * half3(splat.sh10);
//    result += half(SH_C3[2]) * y * (4.0f * zz - xx - yy) * half3(splat.sh11);
//    result += half(SH_C3[3]) * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * half3(splat.sh12);
//    result += half(SH_C3[4]) * x * (4.0f * zz - xx - yy) * half3(splat.sh13);
//    result += half(SH_C3[5]) * z * (xx - yy) * half3(splat.sh14);
//    result += half(SH_C3[6]) * x * (xx - 3.0f * yy) * half3(splat.sh15);
    
    // Convert to valid color range (0-1) and clamp
    return clamp(result , 0.0h, 1.0h);
}

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize) {
    float invViewPosZ = 1 / viewPos.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

    float tanHalfFovX = 1 / projectionMatrix[0][0];
    float tanHalfFovY = 1 / projectionMatrix[1][1];
    float limX = 1.3 * tanHalfFovX;
    float limY = 1.3 * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

    float focalX = screenSize.x * projectionMatrix[0][0] / 2;
    float focalY = screenSize.y * projectionMatrix[1][1] / 2;

    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    float dist = max(0.1, sqrt(mean * mean - det)); // based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu

    // Eigenvalues
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        eigenvector1 = normalize(float2(b, d - lambda2));
    }

    // Gaussian axes are orthogonal
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex) {
    FragmentIn out;

    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;

    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < 0.0 ||
        projectedCenter.z > projectedCenter.w ||
        projectedCenter.x < -bounds ||
        projectedCenter.x > bounds ||
        projectedCenter.y < -bounds ||
        projectedCenter.y > bounds) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[relativeVertexIndex];
    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
    half2 projectedScreenDelta =
        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
        * 2
        * kBoundsRadius
        / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;
    
    // Use spherical harmonics if enabled, otherwise use basic color
    if (uniforms.useSphericalHarmonics) {
        // Calculate view direction for spherical harmonics
        // Use view space position to get direction vector
        float4 worldPos4 = float4(splat.position, 1.0);
        float4 viewPos4 = uniforms.viewMatrix * worldPos4;
        float3 viewDirection = normalize(-viewPos4.xyz);  // Direction from camera to point in view space
        
        // Compute view-dependent color using spherical harmonics
        half3 shColor = computeColorFromSH(splat, viewDirection);
        out.color = half4(shColor, splat.color.a);  // Keep original alpha
    } else {
        // Fallback to basic color (no view-dependent effects)
        out.color = splat.color;
    }
    
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? 0 : exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}
