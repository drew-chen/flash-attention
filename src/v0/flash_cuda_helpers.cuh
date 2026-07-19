#pragma once

#include <cuda_runtime.h>

#include "../cuda_utils.h"
#include "../cuda_utils.cuh"

constexpr int TILE_SZ{16};

/**
Performs a tiled transpose from input[height, width] to output[width, height].
*/
__global__ void transpose(const float *input,
                          float *output,
                          int num_heads,
                          int input_height,
                          int input_width) {
    const auto [batch_idx, head_idx] = get_batch_head_index(num_heads);
    const int matrix_offset =
        batch_head_offset(batch_idx, head_idx, num_heads, input_height * input_width);
    const int input_r = static_cast<int>((blockIdx.y * TILE_SZ) + threadIdx.y);
    const int input_c = static_cast<int>((blockIdx.x * TILE_SZ) + threadIdx.x);

    __shared__ float tile[TILE_SZ][TILE_SZ + 1];
    if (input_r < input_height && input_c < input_width) {
        tile[threadIdx.y][threadIdx.x] = input[matrix_offset + (input_r * input_width) + input_c];
    }
    __syncthreads();

    const int output_c = static_cast<int>((blockIdx.y * TILE_SZ) + threadIdx.x);
    const int output_r = static_cast<int>((blockIdx.x * TILE_SZ) + threadIdx.y);
    if (output_c < input_height && output_r < input_width) {

        output[matrix_offset + (output_r * input_height) + output_c] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void scale(float *data, int num_heads, int data_width, int data_height, float factor) {
    const auto [batch_idx, head_idx] = get_batch_head_index(num_heads);
    const int matrix_offset =
        batch_head_offset(batch_idx, head_idx, num_heads, data_height * data_width);
    const int r = static_cast<int>((blockIdx.y * TILE_SZ) + threadIdx.y);
    const int c = static_cast<int>((blockIdx.x * TILE_SZ) + threadIdx.x);
    if (r >= data_height || c >= data_width) {
        return;
    }

    data[matrix_offset + (r * data_width) + c] *= factor;
}

/**
Traditional matrix multiply where the left matrix's rows are multiplied by
the right matrix's columns.

Naive Algorithm:
A very naive matrix multiply is memory bound.For a thread to calculate an output element
it needs to load in an entire row and column.
Furthermore, adjacent elements of output will load in one of the same row or column.

In terms of arithmetic intensity defined as flops/bytes, this is very poor.

So for NxN matrices, we:
1. load in 2*N elements of 4 bytes each
2. perform N muptlications and N - 1 additions
3. This equates to roughly 1/4 flops/byte

How is this?
Arithmetic intensity = flops/byte = compute throughput/memory bandwith
A 4080 has
* Peak compute throughput: 49 TFLOP/s
* Peak memory bandwidth: 716.8 GB/s

This yields two points of comparison:
1. Given the algorithm’s arithmetic intensity of 0.25 FLOP/byte,
  its memory-bound roofline ceiling is 716.8 × 0.25 = 179.2 GFLOP/s,
  which is about 0.37% of peak FP32 throughput.
2. We can also use the roofline model to visualize.
  The ridge point arithmetic intensity of 68 flops/byte for the 4080 is shown below:

performance
  ^
  |                         compute roof
  |              X-------------------------------
  |             /
  |            /
  |           /
  |          /
  |_________/____________________________________> arithmetic intensity

X is the ridge point with an x coordinate of 68 flops/byte. Given we are below this (to the left),
we are memory bound.

More detail:

X is the minimum arithmetic intensity required to be compute-bound on that hardware
since X = peak throughput FLOP/s / peak bandwidth GB/s = FLOP/B

A < X
means
A < peak compute throughput/ peak memory bandwidth
A*peak memory bandwith < peak compute throughput

Thus, given A is fixed for an algorithm, the compute throughput is memory limited.

If

A*peak memory bandwith > peak compute throughput

We are compute-limited, since the memory-implied performance ceiling is above
the GPU’s peak compute throughput.


Naive tiled algorithm:
This is the basic shared memory tiled implementation, taking advantage of memory
coalescing and reduced global memory traffic. I call it naive still because there
is a long list of further optimizations.

The approach is to before mini matrix multiplies using shared memory and accumulating
the results since summation is associative and commutative.

Suppose the input matrices are the size of a block.
A cuda block will load everything into shared memory.
The advantage of this is re-use:
an output element (i, j) and output element (i, j + 1) both require
the same left matrix row i. Since it's in shared memory, we have effective caching.

To expand this to matrices the size of multiple blocks we simply divide our
input into groups the size of our blocks.
Suppose an input row is 4x the length of a block. We can calculate the output
by performing 4 block-level matrix multiplies, and summing the result of the 4
multiplies into the output.

How is this algorithm?

To compute a block of output (TILE_SZ^2), we load N/TILE_SZ blocks.
The number of flops is the same. Let T = TILE_SZ

Arithmetic intensity = OP/B s
= tile ops / tile memory loaded
# there are N/T iterations, with total 2N multiplies and adds per element across all iterations
# memory loaded is two blocks of block size * num blocks for each iteration
= FLOPs per tile / num iterations * loads per tile per iteration * sizeof(float)
= 2N * T^2 FLOPs / (N/T)(2*T^2)*4B
= 0.25T FLOPs / B
= 4 FLOPs/B

This is a factor of T improvement!
*/
__global__ void matmul(const float *left,
                       const float *right,
                       float *output,
                       int num_heads,
                       int left_height,
                       int shared_dim,
                       int right_width) {
    const auto [batch_idx, head_idx] = get_batch_head_index(num_heads);
    left += batch_head_offset(batch_idx, head_idx, num_heads, left_height * shared_dim);
    right += batch_head_offset(batch_idx, head_idx, num_heads, shared_dim * right_width);
    output += batch_head_offset(batch_idx, head_idx, num_heads, left_height * right_width);
    const int ty = static_cast<int>(threadIdx.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int output_r = (static_cast<int>(blockIdx.y) * TILE_SZ) + ty;
    const int output_c = (static_cast<int>(blockIdx.x) * TILE_SZ) + tx;

    __shared__ float left_tile[TILE_SZ][TILE_SZ];
    __shared__ float right_tile[TILE_SZ][TILE_SZ];

    float dot_prod{};
    // divide left into one row of blocks and divide right into one column of blocks
    for (int i = 0; i < ceil_div(shared_dim, TILE_SZ); i++) {
        // load a tile of left's rows. Use a global row idx to choose the rows,
        // and use local col idx to choose idx since we want load all cols of the rows after the
        // iterations
        const int itr_left_col = (i * TILE_SZ) + tx;
        const int itr_right_row = (i * TILE_SZ) + ty;
        const int left_idx = (output_r * shared_dim) + itr_left_col;
        const int right_idx = (itr_right_row * right_width) + output_c;
        if (output_r < left_height && itr_left_col < shared_dim) {
            left_tile[threadIdx.y][threadIdx.x] = left[left_idx];
        } else {
            left_tile[threadIdx.y][threadIdx.x] = 0;
        }
        // Load a tile of right's cols.
        if (itr_right_row < shared_dim && output_c < right_width) {
            right_tile[threadIdx.y][threadIdx.x] = right[right_idx];
        } else {
            right_tile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int j = 0; j < TILE_SZ; j++) {
            dot_prod += left_tile[threadIdx.y][j] * right_tile[j][threadIdx.x];
        }
    }
    if (output_r >= left_height || output_c >= right_width) {
        return;
    }
    output[(output_r * right_width) + output_c] = dot_prod;
}
