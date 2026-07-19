#pragma once

#include <cuda/std/concepts>
#include <cuda_runtime.h>
#include <type_traits>

#include "../cuda_utils.cuh"
/*
Approach 1
1. reduce to find max(x)
2. map to e^(xi - max(x))
3. find the sum
4. map to softmax

Fused approach

1. reduce to find max(x)
2. transform AND find sum e^(xi - max(x))
3. map to softmax

*/

namespace flash_attention::detail {

enum class ReductionOp { SUM, MAX };

__device__ __forceinline__ float gpu_add(float a, float b) { return a + b; }
__device__ __forceinline__ double gpu_add(double a, double b) { return a + b; }

__device__ __forceinline__ float gpu_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double gpu_max(double a, double b) { return fmax(a, b); }

__device__ __forceinline__ float atomicMaxFloat(float *address, float val) {
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        // Compare current value with val and find the max
        float max_val = gpu_max(val, __int_as_float(assumed));

        // Attempt to swap. atomicCAS returns the value that was actually
        // at the address. If it matches 'assumed', the swap succeeded.
        old = atomicCAS(address_as_i, assumed, __float_as_int(max_val));

    } while (assumed != old);

    return __int_as_float(old);
}

template <typename F, typename T>
concept Mapper = requires(F f, T val) {
    { f(val) } -> cuda::std::same_as<T>;
};

// use this so functions don't need to keep forwarding template args
template <typename T, ReductionOp Op> struct ReductionPolicy {
    // Identity values for initialization
    static __device__ __forceinline__ T identity() {
        if constexpr (Op == ReductionOp::SUM)
            return static_cast<T>(0);
        if constexpr (Op == ReductionOp::MAX)
            return static_cast<T>(-1e38); // simplified -inf
    }

    // Mutates and reduces val into the accumulator, and returns the mapped val
    static __device__ __forceinline__ void apply_op(T &accumulator, const T val) {
        accumulator = reduce(accumulator, val);
    }

    // Used to atomically combine the result of a reduction kernel
    static __device__ __forceinline__ void apply_atomic(T *result, const T val) {
        if constexpr (Op == ReductionOp::SUM)
            atomicAdd(result, val);
        else if constexpr (Op == ReductionOp::MAX)
            atomicMaxFloat(result, val);
    }

    static __device__ __forceinline__ void warp_reduce(T &val) {
        unsigned int mask = __activemask();
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            apply_op(val, __shfl_down_sync(mask, val, offset));
        }
    }

  private:
    static __device__ __forceinline__ T reduce(const T accumulator, const T val) {
        if constexpr (Op == ReductionOp::SUM)
            return gpu_add(accumulator, val);
        if constexpr (Op == ReductionOp::MAX)
            return gpu_max(accumulator, val);
    }
};

template <typename T> struct MapIdentity {
    // RVO: return value optimization
    __device__ __forceinline__ T operator()(const T &val) const { return val; }
};

template <typename T> struct MapExpMinus {
    const __restrict__ T *max_ptr;
    int num_heads;
    int rows;
    // RVO: return value optimization
    __device__ __forceinline__ T operator()(const T &val) const {
        // ldg forces to read from the cache
        const auto [batch_idx, head_idx] = flash_attention::detail::get_batch_head_index(num_heads);
        const int stat_offset =
            flash_attention::detail::batch_head_offset(batch_idx, head_idx, num_heads, rows) +
            blockIdx.y;
        return __expf(val - __ldg(max_ptr + stat_offset));
    }
};

template <typename T> struct MapDivision {
    const __restrict__ T *divisor_ptr;
    int num_heads;
    int rows;
    // RVO: return value optimization
    __device__ __forceinline__ T operator()(const T &val) const {
        const auto [batch_idx, head_idx] = flash_attention::detail::get_batch_head_index(num_heads);
        const int stat_offset =
            flash_attention::detail::batch_head_offset(batch_idx, head_idx, num_heads, rows) +
            blockIdx.y;
        return val / __ldg(divisor_ptr + stat_offset);
    }
};

// mapper is last param since defaulted. mutates input and reduces
// this map reduce which modifies the output with the mapping and returns the reduced value
// exception: if the mapping is the identity, only the reduction is performed
template <typename T,
          ReductionOp Op,
          unsigned int BlockSize,
          int CoarsenFactor,
          Mapper<T> M = MapIdentity<T>>
__global__ void map_reduce_kernel(const T *__restrict__ input,
                                  T *__restrict__ mapped_output,
                                  T *__restrict__ reduction_output,
                                  const int num_heads,
                                  const int rows,
                                  const int head_dim,
                                  // all args by default are passed in const memory
                                  // this is a compiler hint that this will never chang ethroughout
                                  // lifetime of program
                                  // __grid_constant__ ensures all threads use the same single
                                  // address in constant memory, saving significant memory and
                                  // improving performance.
                                  const __grid_constant__ M mapper = M{}) {

    // cannot use auto here
    __shared__ T sdata[BlockSize];

    // Alias map reducer for cleaner syntax
    using R = ReductionPolicy<T, Op>;

    auto accumulator = R::identity();
    int tid = threadIdx.x;
    const auto [batch_idx, head_idx] = flash_attention::detail::get_batch_head_index(num_heads);
    const int batch_head_offset_elements =
        flash_attention::detail::batch_head_offset(batch_idx, head_idx, num_heads,
                                                   rows * head_dim);
    const int row_offset = batch_head_offset_elements + blockIdx.y * head_dim;
    const int idx = blockIdx.x * (BlockSize * CoarsenFactor) + tid;

// 1. Loop Coarsening & mapping to output
/*
Note: N <= 500,000, so we do not need to do a grid stride since max grid size is 2^31 - 1. We can
assume # elements processed = block # of threads * coarsen To coarsen, we have to repeat the
gathering of input elements by the # of times we coarsen To get our block to read independent
elements, we conceptually divide the input into multiple blocks of elements per block of threads. So
with coarsen = 4, we iterate over 4 blocks of data with our 1 block of threads
*/
#pragma unroll
    for (int c = 0; c < CoarsenFactor; ++c) {
        int coarsen_i = idx + c * BlockSize;
        if (coarsen_i < head_dim) {
            const int row_i = row_offset + coarsen_i;
            T mapped_val = mapper(input[row_i]);
            R::apply_op(accumulator, mapped_val);
            if constexpr (!std::is_same_v<M, MapIdentity<T>>) {
                mapped_output[row_i] = mapped_val;
            }
        }
    }

    sdata[tid] = accumulator;
    __syncthreads();
    // 2. Shared Memory Tree Reduction
    // No more mapping needed
    for (unsigned int stride = BlockSize / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            R::apply_op(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    // 3. Final Warp Hand-off
    if (tid < 32) {
        T val = sdata[tid];
        R::warp_reduce(val);
        if (tid == 0) {
            const int stat_offset =
                flash_attention::detail::batch_head_offset(batch_idx, head_idx, num_heads, rows) +
                blockIdx.y;
            R::apply_atomic(reduction_output + stat_offset, val);
        }
    }
}

template <typename T, Mapper<T> M>
__global__ void map_kernel(const T *input,
                           T *__restrict__ output,
                           const int num_heads,
                           const int rows,
                           const int head_dim,
                           const __grid_constant__ M mapper) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= head_dim)
        return;
    const auto [batch_idx, head_idx] = flash_attention::detail::get_batch_head_index(num_heads);
    const int batch_head_offset_elements =
        flash_attention::detail::batch_head_offset(batch_idx, head_idx, num_heads,
                                                   rows * head_dim);
    const int i = batch_head_offset_elements + blockIdx.y * head_dim + col;
    output[i] = mapper(input[i]);
};

__global__ void initialize_max_kernel(float *output, int N) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        output[row] = -1e38F;
    }
}

// V0 host launcher for row-wise softmax over a contiguous
// [batch_size, num_heads, rows, head_dim] tensor.
// The three kernel launches provide the global synchronization points between
// max reduction, exponentiation/sum reduction, and normalization.
inline void softmax_rows_launch(const float *input,
                                float *output,
                                int batch_size,
                                int num_heads,
                                int rows,
                                int head_dim) {
    const int mr_block_size = 256;
    const int coarsen = 4;
    const int elements_per_block = mr_block_size * coarsen;
    const int mr_grid_dim = (head_dim - 1) / elements_per_block + 1;
    const int batch_head_count = batch_size * num_heads;
    const int stat_count = batch_head_count * rows;

    float *device_stats;
    cudaMalloc(&device_stats, 2 * static_cast<std::size_t>(stat_count) * sizeof(float));
    float *device_exp_sum_ptr = device_stats;
    float *device_max_ptr = device_stats + stat_count;
    cudaMemset(device_exp_sum_ptr, 0, static_cast<std::size_t>(stat_count) * sizeof(float));
    const int init_block_size = 256;
    const int init_grid_dim = (stat_count - 1) / init_block_size + 1;
    initialize_max_kernel<<<init_grid_dim, init_block_size>>>(device_max_ptr, stat_count);

    const dim3 reduce_grid{static_cast<unsigned int>(mr_grid_dim), static_cast<unsigned int>(rows),
                           static_cast<unsigned int>(batch_head_count)};
    map_reduce_kernel<float, ReductionOp::MAX, mr_block_size, coarsen>
        <<<reduce_grid, mr_block_size>>>(input, nullptr, device_max_ptr, num_heads, rows, head_dim);
    map_reduce_kernel<float, ReductionOp::SUM, mr_block_size, coarsen, MapExpMinus<float>>
        <<<reduce_grid, mr_block_size>>>(input, output, device_exp_sum_ptr, num_heads, rows,
                                         head_dim,
                                         MapExpMinus<float>{device_max_ptr, num_heads, rows});

    const int map_block_size = 1024;
    const int map_grid_dim = (head_dim - 1) / map_block_size + 1;
    const dim3 map_grid{static_cast<unsigned int>(map_grid_dim), static_cast<unsigned int>(rows),
                        static_cast<unsigned int>(batch_head_count)};
    map_kernel<<<map_grid, map_block_size>>>(output, output, num_heads, rows, head_dim,
                                             MapDivision<float>{device_exp_sum_ptr, num_heads,
                                                                rows});

    cudaFree(device_stats);
}

}  // namespace flash_attention::detail
