#include <metal_stdlib>
#include "ShaderCommon.h"

using namespace metal;

// ============================================================================
// COMMON STRUCTURES
// ============================================================================

// Structure for sorting: pairs depth with original index
struct SplatDepthKey {
    float depth;
    uint32_t index;
};

// Uniforms for depth computation
struct DepthComputeUniforms {
    float3 cameraPosition;
    float3 cameraForward;
    uint32_t splatCount;
    bool sortByDistance;  // true = euclidean distance, false = dot product with forward
};

// Uniforms for sorting passes
struct SortUniforms {
    uint32_t count;           // Total number of elements
    uint32_t stageDistance;   // Distance between compared elements (for bitonic)
    uint32_t passDistance;    // Sub-pass distance (for bitonic)
    uint32_t ascending;       // Sort direction (1 = ascending, 0 = descending)
};

// ============================================================================
// DEPTH COMPUTATION KERNELS
// ============================================================================

// Compute depth values for all splats (ascending order)
kernel void computeSplatDepths(
    constant Splat* splats [[buffer(0)]],
    device float* depths [[buffer(1)]],
    device uint32_t* indices [[buffer(2)]],
    constant DepthComputeUniforms& uniforms [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.splatCount) return;

    float3 splatPosition = float3(splats[id].position);

    float depth;
    if (uniforms.sortByDistance) {
        float3 delta = splatPosition - uniforms.cameraPosition;
        depth = dot(delta, delta);  // length squared
    } else {
        depth = dot(splatPosition, uniforms.cameraForward);
    }

    depths[id] = depth;
    indices[id] = id;
}

// Compute negated depth for descending sort
kernel void computeSplatDepthsDescending(
    constant Splat* splats [[buffer(0)]],
    device float* depths [[buffer(1)]],
    device uint32_t* indices [[buffer(2)]],
    constant DepthComputeUniforms& uniforms [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.splatCount) return;

    float3 splatPosition = float3(splats[id].position);

    float depth;
    if (uniforms.sortByDistance) {
        float3 delta = splatPosition - uniforms.cameraPosition;
        depth = dot(delta, delta);
    } else {
        depth = dot(splatPosition, uniforms.cameraForward);
    }

    // Negate for descending order (sort ascending gives far to near)
    depths[id] = -depth;
    indices[id] = id;
}

// ============================================================================
// BITONIC SORT
// ============================================================================
// Bitonic sort is highly parallel - all comparisons in each pass are independent.
// Complexity: O(n * log²n) comparisons, but O(log²n) sequential passes
// Each pass can process all n elements in parallel.
//
// How it works:
// 1. Build bitonic sequences of increasing size (2, 4, 8, 16, ...)
// 2. A bitonic sequence goes up then down (or down then up)
// 3. Merge bitonic sequences by comparing elements at specific distances
// 4. After log²n passes, the entire array is sorted

// Single pass of bitonic sort - compares and swaps elements
// Fixed version with proper handling for non-power-of-2 sizes
kernel void bitonicSortPass(
    device float* keys [[buffer(0)]],
    device uint32_t* values [[buffer(1)]],
    constant SortUniforms& uniforms [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint count = uniforms.count;
    uint stageDistance = uniforms.stageDistance;  // 2^stage (block size for direction)
    uint passDistance = uniforms.passDistance;    // Distance between compared elements

    if (id >= count) return;

    // Find the partner element to compare with using XOR
    uint partnerId = id ^ passDistance;

    // Only the thread with smaller index handles the comparison
    // to avoid double-swapping
    if (partnerId <= id) return;

    // Determine sort direction for this pair
    // Use bitwise AND for correct bitonic direction calculation
    // (id & stageDistance) == 0 means ascending, else descending
    bool ascending = (id & stageDistance) == 0;
    if (uniforms.ascending == 0) ascending = !ascending;  // Flip if descending overall

    // Handle out-of-bounds partner (treat as +∞ for ascending, -∞ for descending)
    if (partnerId >= count) {
        // Partner is out of bounds
        // For ascending: out-of-bounds is +∞, so no swap needed (current is already smaller)
        // For descending: out-of-bounds is -∞, so swap if current > -∞ (always true)
        // But since we can't actually swap with nothing, we just skip
        return;
    }

    float keyA = keys[id];
    float keyB = keys[partnerId];

    // Handle NaN values: treat NaN as +∞ for ascending, -∞ for descending
    // This ensures NaN values sort to the end and don't corrupt the sort
    bool aIsNaN = isnan(keyA);
    bool bIsNaN = isnan(keyB);

    bool shouldSwap;
    if (aIsNaN && bIsNaN) {
        shouldSwap = false;  // Both NaN, no swap needed
    } else if (aIsNaN) {
        // A is NaN: for ascending, NaN should go to end (higher index), so swap if at lower index
        shouldSwap = ascending;  // Ascending: swap NaN to higher index
    } else if (bIsNaN) {
        // B is NaN: for ascending, NaN should go to end (higher index), so no swap
        shouldSwap = !ascending;  // Ascending: don't swap, keep NaN at higher index
    } else {
        // Normal comparison
        // For ascending: swap if keyA > keyB (put smaller at lower index)
        // For descending: swap if keyA < keyB (put larger at lower index)
        shouldSwap = ascending ? (keyA > keyB) : (keyA < keyB);
    }

    if (shouldSwap) {
        uint32_t valA = values[id];
        uint32_t valB = values[partnerId];

        keys[id] = keyB;
        keys[partnerId] = keyA;
        values[id] = valB;
        values[partnerId] = valA;
    }
}

// Local bitonic sort within a threadgroup (faster for small chunks)
// Uses threadgroup memory for better performance
kernel void bitonicSortLocal(
    device float* keys [[buffer(0)]],
    device uint32_t* values [[buffer(1)]],
    constant SortUniforms& uniforms [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]]
) {
    // Shared memory for local sorting
    threadgroup float localKeys[1024];
    threadgroup uint32_t localValues[1024];

    uint count = uniforms.count;
    uint globalId = id;

    // Load into shared memory
    if (globalId < count) {
        localKeys[lid] = keys[globalId];
        localValues[lid] = values[globalId];
    } else {
        localKeys[lid] = INFINITY;  // Padding with max value
        localValues[lid] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform bitonic sort within threadgroup
    for (uint stage = 1; stage <= 10; stage++) {  // Up to 2^10 = 1024 elements
        uint stageDistance = 1u << stage;
        if (stageDistance > groupSize) break;

        for (uint pass = stage; pass > 0; pass--) {
            uint passDistance = 1u << (pass - 1);

            uint partnerId = lid ^ passDistance;

            if (partnerId > lid && partnerId < groupSize) {
                bool ascending = ((lid / stageDistance) % 2) == 0;
                if (uniforms.ascending == 0) ascending = !ascending;

                float keyA = localKeys[lid];
                float keyB = localKeys[partnerId];

                bool shouldSwap = ascending ? (keyA > keyB) : (keyA < keyB);

                if (shouldSwap) {
                    // Swap
                    float tempKey = localKeys[lid];
                    uint32_t tempVal = localValues[lid];
                    localKeys[lid] = localKeys[partnerId];
                    localValues[lid] = localValues[partnerId];
                    localKeys[partnerId] = tempKey;
                    localValues[partnerId] = tempVal;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back to global memory
    if (globalId < count) {
        keys[globalId] = localKeys[lid];
        values[globalId] = localValues[lid];
    }
}

// ============================================================================
// RADIX SORT
// ============================================================================
// Radix sort processes numbers digit by digit (or bit by bit).
// Complexity: O(n * k) where k = number of bits/digits
// GPU implementation is trickier due to counting/prefix-sum dependencies.
//
// How it works:
// 1. For each bit position (LSB to MSB):
//    a. Count how many 0s and 1s
//    b. Compute prefix sums (exclusive scan)
//    c. Scatter elements to new positions
// 2. After processing all bits, array is sorted

// Count bits for radix sort histogram
// Each threadgroup counts bits for its chunk, then we do a global prefix sum
kernel void radixCountBits(
    device float* keys [[buffer(0)]],
    device uint32_t* histogram [[buffer(1)]],  // [numGroups * 2] for 0s and 1s count
    constant SortUniforms& uniforms [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint localCount0;
    threadgroup atomic_uint localCount1;

    if (lid == 0) {
        atomic_store_explicit(&localCount0, 0, memory_order_relaxed);
        atomic_store_explicit(&localCount1, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint count = uniforms.count;
    uint bitPosition = uniforms.passDistance;  // Which bit we're sorting by

    if (id < count) {
        // Reinterpret float as uint for bit manipulation
        // Note: For proper float sorting, we need to handle sign bit
        float key = keys[id];
        uint32_t keyBits = as_type<uint32_t>(key);

        // Handle negative floats: flip all bits if negative, else flip sign bit
        // This ensures correct ordering: -inf < negative < 0 < positive < +inf
        if (keyBits & 0x80000000) {
            keyBits = ~keyBits;  // Negative: flip all bits
        } else {
            keyBits ^= 0x80000000;  // Positive: flip sign bit
        }

        uint bit = (keyBits >> bitPosition) & 1;

        if (bit == 0) {
            atomic_fetch_add_explicit(&localCount0, 1, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&localCount1, 1, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write counts to global histogram
    if (lid == 0) {
        histogram[groupId * 2 + 0] = atomic_load_explicit(&localCount0, memory_order_relaxed);
        histogram[groupId * 2 + 1] = atomic_load_explicit(&localCount1, memory_order_relaxed);
    }
}

// Scatter elements to sorted positions based on prefix sums
kernel void radixScatter(
    device float* keysIn [[buffer(0)]],
    device float* keysOut [[buffer(1)]],
    device uint32_t* valuesIn [[buffer(2)]],
    device uint32_t* valuesOut [[buffer(3)]],
    device uint32_t* prefixSums [[buffer(4)]],  // Global prefix sums
    constant SortUniforms& uniforms [[buffer(5)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]]
) {
    // Local prefix sums for this threadgroup
    threadgroup uint localPrefix0[1024];
    threadgroup uint localPrefix1[1024];

    uint count = uniforms.count;
    uint bitPosition = uniforms.passDistance;

    if (id >= count) return;

    float key = keysIn[id];
    uint32_t value = valuesIn[id];
    uint32_t keyBits = as_type<uint32_t>(key);

    // Handle float ordering
    if (keyBits & 0x80000000) {
        keyBits = ~keyBits;
    } else {
        keyBits ^= 0x80000000;
    }

    uint bit = (keyBits >> bitPosition) & 1;

    // Get global offset from prefix sums
    uint globalOffset0 = prefixSums[groupId * 2 + 0];
    uint globalOffset1 = prefixSums[groupId * 2 + 1];

    // Compute local offset within threadgroup (simplified - actual impl needs local scan)
    // This is a simplified version; production code would use parallel prefix sum

    uint destIndex;
    if (bit == 0) {
        destIndex = globalOffset0 + lid;  // Simplified
    } else {
        destIndex = globalOffset1 + lid;  // Simplified
    }

    if (destIndex < count) {
        keysOut[destIndex] = key;
        valuesOut[destIndex] = value;
    }
}

// ============================================================================
// REORDERING KERNEL
// ============================================================================

// Reorder splats based on sorted indices
kernel void reorderSplats(
    constant Splat* srcSplats [[buffer(0)]],
    device Splat* dstSplats [[buffer(1)]],
    constant uint32_t* sortedIndices [[buffer(2)]],
    constant uint32_t& splatCount [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= splatCount) return;

    uint32_t srcIndex = sortedIndices[id];
    dstSplats[id] = srcSplats[srcIndex];
}
