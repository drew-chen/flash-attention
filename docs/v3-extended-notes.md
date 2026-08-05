# V3 extended notes

These notes describe my simplified implementation of Tri Dao's flash attention 2.

flash_forward_v3_cuda_launch expects raw pointers for tensors shaped as:

- q: [B, H, M, D]
- k: [B, H, N, D]
- v: [B, H, N, D]

- out: [B, H, M, D]

  For each batch element b and head h, out[b, h] =
  softmax((q[b, h] * k[b, h]^T) / sqrt(D)) * v[b, h].

Dimensions:

- (batch_size) B: batch size. How many independent sequences are processed together.
- (num_heads) H: number of attention heads per sequence.
- (query_seq_len) M: number of query/output rows.
- (kv_seq_len) N: number of key/value rows. Self-attention uses M = N, while
  cross-attention may use different lengths.
- (head_dim) D: head dimension. Size of the per-token vector inside one head.

## V3 implementation

This implements the warp partitioning described by FA2.

V3 starts as a working copy of V2 and follows FA2's warp ownership more closely,
without tensor cores or an asynchronous pipeline. The main goal is to fit at
least two blocks per SM on the RTX 4080.

1. We chose D = 64 for convenience so the warp-local state can use simple
   compile-time arrays. Use 256 threads (eight warps) per block; with B_r = 64,
   each warp is assigned eight rows from the Q/output tile.

2. Keep the running maximum, denominator, and unnormalized output in the owning
   warp's registers. Lanes cooperatively calculate QK and PV for those rows, so
   no inter-warp reduction is needed.

3. Keep Q in shared memory for the full K/V loop, reuse one shared buffer for
   K then V, and keep the unnormalized P tile in shared memory for the PV step.

4. Preserve V2's query-tile grid parallelism, delayed output normalization,
   cross-attention support, and partial-tile bounds handling.

## Algorithm

```text
Initialize:
    B_c = 32    # Number of score columns and K/V rows
                # processed per data tile (can be tuned).
    B_r = 64    # Number of score rows and Q rows
                # processed per data tile (can be tuned).

    WARP_SIZE = 32
    NUM_WARPS = 8
    WARP_TILE_ROWS = B_r / NUM_WARPS

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks

warp = threadIdx.x / WARP_SIZE
lane = threadIdx.x % WARP_SIZE
warp_row_start = warp * WARP_TILE_ROWS

    # Each thread block processes a Q slice with shape up to (B_r x D) and
    # produces the matching output rows. Its warps partition those rows, with
    # each warp assigned WARP_TILE_ROWS contiguous rows.
    #
    # Warp-local ownership and distribution:
    #
    # State         | Logical shape per warp | Distribution across lanes      | Per-lane storage
    # --------------|------------------------|--------------------------------|-----------------
    # Scores S      | (WARP_TILE_ROWS x B_c) | One K row per lane             | (WARP_TILE_ROWS)
    # Maximum M     | (WARP_TILE_ROWS)       | Replicated in every lane       | (WARP_TILE_ROWS)
    # Denominator L | (WARP_TILE_ROWS)       | Replicated in every lane       | (WARP_TILE_ROWS)
    # Output O      | (WARP_TILE_ROWS x D)   | D/32=2 output indices per lane | Two (WARP_TILE_ROWS)

wM_i_replicated = fill(WARP_TILE_ROWS, -infinity)
wL_i_replicated = zeros(WARP_TILE_ROWS)

    # Each has shape (WARP_TILE_ROWS). Every lane holds the same running max and
    # denominator for the rows assigned to its warp.

wO_i_left_unnormalized = zeros(WARP_TILE_ROWS)
wO_i_right_unnormalized = zeros(WARP_TILE_ROWS)

    # D=64 is twice the 32-lane warp size, so each lane accumulates two output
    # dimensions for every row assigned to its warp. Rather than making this a
    # 2D wO, I've chosen to just use two separate arrays for performance
    # (less loops needed) and simplicity.
    # These arrays correspond to the two iterations of:
    #     for output_dim = lane; output_dim < D; output_dim += WARP_SIZE
    # The left array holds the output dimension indexed by lane, and the right
    # holds the dimension indexed by lane + WARP_SIZE. Together, the lanes cover
    # all D output dimensions without an inter-lane reduction.

# -- Load a Q tile with shape up to (B_r x D) into shared memory --

sQ_i = Q[b, h, i*B_r: min((i + 1)*B_r, M), :]

    # (Up to B_r x D): Stage the block's Q rows in shared memory.

for each K/V block j = 0 to T_c - 1:

    sK_j = K[b, h, j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Stage up to B_c K rows in shared memory.


    wS_ij = sQ_i[warp_row_start : warp_row_start + WARP_TILE_ROWS, :]
              @ sK_j[lane, :] * SOFTMAX_SCALE

        # (WARP_TILE_ROWS): Each lane computes one score for each of its warp's
        # Q rows against global K row j*B_c + lane. Together, all lanes compute
        # a (WARP_TILE_ROWS x B_c) score fragment. Invalid rows are masked.

    key_tile_row_max = warp_allreduce_max(wS_ij)

        # (WARP_TILE_ROWS): Reduce corresponding lane values to get one maximum
        # per warp-owned Q row, replicated in every lane.

    wM_i_new_replicated = max(wM_i_replicated, key_tile_row_max)

    p = exp(wS_ij - wM_i_new_replicated)
    sP_ij_unnormalized[warp_row_start : warp_row_start + WARP_TILE_ROWS, lane] = p

        # Each lane writes one P value per warp-owned Q row. Together, the
        # block materializes the (B_r x B_c) P tile in shared memory.

    key_tile_row_sum = warp_allreduce_sum(p)

        # (WARP_TILE_ROWS): One sum per warp-owned Q row, replicated in every
        # lane.

    old_scale = exp(wM_i_replicated - wM_i_new_replicated)

    wL_i_replicated = old_scale * wL_i_replicated + key_tile_row_sum
    wO_i_left_unnormalized *= old_scale
    wO_i_right_unnormalized *= old_scale
    wM_i_replicated = wM_i_new_replicated

        # Update the online-softmax state and rescale the running output to the
        # new maximum before adding the current PV contribution.

    # -- Compute running output --

    sV_j = V[b, h, j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Stage up to B_c V rows. This reuses the shared storage
        # previously occupied by sK_j.

    wO_i_left_unnormalized +=
        sP_ij_unnormalized[warp_row_start : warp_row_start + WARP_TILE_ROWS, :]
            @ sV_j[:, lane]
    wO_i_right_unnormalized +=
        sP_ij_unnormalized[warp_row_start : warp_row_start + WARP_TILE_ROWS, :]
            @ sV_j[:, lane + WARP_SIZE]

        # Each lane accumulates two (WARP_TILE_ROWS) output vectors. Together,
        # the warp calculates all D=64 output dimensions for its rows.

# Apply the softmax denominator once after processing every K/V tile.
O[b, h, i*B_r + warp_row_start : i*B_r + warp_row_start + WARP_TILE_ROWS, lane] =
    wO_i_left_unnormalized / wL_i_replicated

O[b, h, i*B_r + warp_row_start : i*B_r + warp_row_start + WARP_TILE_ROWS,
  lane + WARP_SIZE] =
    wO_i_right_unnormalized / wL_i_replicated

    # Each lane normalizes and stores its two output dimensions. Stores outside
    # the final partial Q tile are skipped.
```

## Source mapping

The CUDA implementation is in
[`src/v3/flash_kernel.cu`](../src/v3/flash_kernel.cu). Other head dimensions
are redirected to V2 by the C++ binding.

Variables follow the FA1 paper's notation: `s` prefixes shared-memory
pointers, `g` prefixes global-memory pointers, and `_i` denotes a subscript
of `i`.
