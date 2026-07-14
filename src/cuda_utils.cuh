#pragma once

#include <cuda_runtime.h>

struct BatchHeadIndex {
    int batch_idx;
    int head_idx;
};

__device__ inline BatchHeadIndex get_batch_head_index(int num_heads) {
    const int bh_idx = static_cast<int>((blockIdx.z * blockDim.z) + threadIdx.z);
    return BatchHeadIndex{bh_idx / num_heads, bh_idx % num_heads};
}

__device__ inline int batch_head_offset(int batch_idx,
                                        int head_idx,
                                        int num_heads,
                                        int elements_per_batch_head) {
    return ((batch_idx * num_heads) + head_idx) * elements_per_batch_head;
}
