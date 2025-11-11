//
//  Sort.metal
//  MetalSplatter
//
//  Created by Sudeep Sharma on 11/11/2025.
//

#include <metal_stdlib>
using namespace metal;

struct SplatIndexAndDepth {
    uint index;
    float depth;
};

// Bitonic sort for a power of 2
kernel void bitonicSort(device SplatIndexAndDepth *data,
                        uint stage,
                        uint passOfStage,
                        uint direction,
                        const device uint &count) {
    uint threadID = uint(gl_GlobalInvocationID.x);

    if (threadID < count / 2) {
        uint i = threadID;
        uint j = i + (1 << passOfStage);

        // Calculate the indices to compare and swap
        uint group = i / (1 << stage);
        uint inGroupOffset = i % (1 << stage);
        uint pairOffset = 1 << passOfStage;
        uint index1 = (group * (1 << (stage + 1))) + inGroupOffset;
        uint index2 = index1 + pairOffset;

        if (index2 < count) {
            device SplatIndexAndDepth &item1 = data[index1];
            device SplatIndexAndDepth &item2 = data[index2];

            bool shouldSwap = (item1.depth < item2.depth);
            if ((direction == 1 && shouldSwap) || (direction == 0 && !shouldSwap)) {
                swap(item1, item2);
            }
        }
    }
}
