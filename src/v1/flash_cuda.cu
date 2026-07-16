#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

struct FlashForwardKernelParams {
    const float *q;
    const float *k;
    const float *v;
    float *const out;
    // global memory ptr allocated for a draft v0 implementation
    float *const intermediate_score;
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
};
