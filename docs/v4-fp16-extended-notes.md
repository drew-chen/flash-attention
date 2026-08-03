# V4 FP16 extended notes - What are tensor cores and how can they be used here?

## Tensor cores

Tensor cores perform warp-level matrix multiply-accumulate operations:

```text
D = A @ B + C
```

The operand and accumulator fragments are distributed across the registers of
an entire warp.

Two common ways to access tensor cores directly from a CUDA kernel are the
higher-level C++ WMMA API and lower-level PTX `mma.sync` instructions. WMMA
intentionally hides the mapping between logical matrix elements and each
lane's registers. This provides a portable and convenient interface, but makes
coordinate-dependent operations, such as row-wise softmax scaling, difficult
to perform directly on the fragments. A common solution is to store the
fragment into shared memory in a conventional matrix layout.

PTX MMA is more cumbersome, but exposes a documented lane-to-register layout
for each instruction shape. This can allow subsequent operations to consume or
modify accumulator values directly in registers, avoiding an intermediate
shared-memory round trip. It does not eliminate shared memory generally;
shared memory is still commonly used to stage MMA operands.


## Tensor core usage

As tensor cores are designed for FMAs, the two places they can be used are the
S = Q @ K^T and O = P @ V within a block's calculation.

WMMA can be used for both, but v3 and v4 of my algorithm, which is based on
FA2, rely on warp-level ownership of output to keep data independent between
warps and resident in registers as much as possible, reducing shared memory usage.
WMMA is counter-productive here, because matmuls will to be materialized in shared memory
only to be loaded back into registers.

For O = P@V, this is not a good idea because O is output that is read and written
after each iteration over K/V, and benefits greatly from being kept in registers.
For S = Q @ K^T this can potentially work because S is not built upon, meaning
it only needs to be read once and the rest of the attention steps can be done in
registers.

To use WMMA, we mainly need to think about the shape of the multiply. We
already



[PTX ISA: Matrix Fragments for `mma.m16n8k16`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)

[CUDA C++ Programming Guide: Element Types and Matrix Sizes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes)

## Algorithm

P@V and Q@K^T within a block's calculation will be done using tensor cores.
MMA matmul: P @ V
WMMA matmul: Q @ K^T



```text

Initialize:
    B_c = 32    # Number of score cols and K/V rows
                # processed per data tile (can be tuned).
    B_r = 64    # Number of score rows and Q rows
                # processed per data tile (can be tuned).

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks

    sS_ij = wmma::load_matrix_sync #

    # This represents conceptual state for one Q tile. Each warp owns the
    # entries corresponding to one or more complete query rows.

    wM_i_replicated = fill(B_r, -infinity) # (B_r): Running maximum state.
    wL_i_replicated = zeros(B_r)           # (B_r): Running softmax denominator.
    wO_i_unnormalized = zeros(B_r, D)      # (B_r x D): Running output numerator.



# -- Load shared memory 2D tile dim (B_r x D) --

s_Q_i = Q[b, h, i*B_r: min((i + 1)*B_r, M), :]


for each K/V block j = 0 to T_c - 1:

    s_K_j = K[b, h, j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Stage up to B_c K rows in shared memory.


    wS_ij = s_Q_i @ s_K_j^T / sqrt(D)

        # Each warp owns complete score rows. Each lane stores one score
        # column for its rows, so the complete score tile is register-local.


    wM_i_new = max(wM_i_replicated, rowmax(wS_ij))

        # (B_r). Each warp calculates the entries for its owned rows, then
        # replicates the result across its lanes with warp shuffles.


    sP_ij_unnormalized = exp(wS_ij - wM_i_new)

        # (B_r x B_c): Calculate the attention-weight numerator using the
        # updated max, but do not apply the softmax denominator. V3 stores
        # this intermediate in shared memory.


    wM_i_rescale_factor = exp(wM_i_replicated - wM_i_new)

        # (B_r): By multiplying by this constant, the exp scale of the previous
        iteration is updated to this iteration's max.

    wL_i_replicated = wM_i_rescale_factor * wL_i_replicated
                        + rowsum(sP_ij_unnormalized)

        # (B_r): Update running denominator. By the end of the algorithm,
        # wL_i_replicated is the denominator for each score row.

    # -- Compute running output --

    # sV_j reuses the shared storage previously occupied by sK_j.
    s_V_j = V[b, h, j*B_c: min((j + 1)*B_c, N), :]

    wO_i_unnormalized = wO_i_unnormalized * wM_i_rescale_factor
                          + sP_ij_unnormalized @ s_V_j

        # (B_r x D): Rescale softmax numerator for running output then
        # add this tile to the running output.

    wM_i_replicated = wM_i_new

# Apply the softmax denominator once after processing every K/V tile.
O[b, h, i*B_r: min((i + 1)*B_r, M), :] = wO_i_unnormalized / wL_i_replicated

```

---

In the following impl, variables are named similar to the FA1 paper
and s prefix means shared mem ptr, g prefix means global mem ptr,
and finally _i means a subscript of i.
