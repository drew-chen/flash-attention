# V2 extended notes

These notes describe my simplified implementation of FlashAttention-2.

`flash_forward_v2_cuda_launch` expects raw pointers for tensors shaped as:

- q: [B, H, M, D]
- k: [B, H, N, D]
- v: [B, H, N, D]

- out: [B, H, M, D]

  For each batch element b and head h, out[b, h] =
  softmax((q[b, h] * k[b, h]^T) / sqrt(D)) * v[b, h].

Dimensions:

- (batch_size) B: batch size. How many independent sequences are processed together.
- (num_heads) H: number of attention heads per sequence.
- (query_seq_len) M: query sequence length.
- (kv_seq_len) N: key/value sequence length, also called context length.
- (head_dim) D: head dimension. Size of the per-token vector inside one head.

Q, K, and V must be contiguous CUDA float32 tensors with matching batch, head,
and head-dimension sizes. K and V must have identical shapes; M and N may differ.

## FlashAttention-2

### 1. Delay normalizing O by the softmax denominator

Rather than dividing partial outputs by the updated softmax denominator l_i after
each K/V tile, store the unnormalized numerator and apply the denominator once at
the end, prior to writing the final output for a block. The paper calls this output
"unscaled", though technically the old numerator is still rescaled by
exp(m_old - m_new) whenever the running maximum changes.

### 2. Parallelize blocks on the query sequence dimension

As seen from the v1 profiling, the v1 algorithm can have poor performance
as sequence length grows and the batch size decreases. This is addressed
by dividing the work for a particular [B, H] across multiple query-tile blocks.
In this implementation, the query-tile index is represented by the grid's z dim.

### 3. Better warp partitioning

This section describes the original optimized FA1 CUDA kernel's warp partitioning,
as described by the FA2 paper, vs. FA2's warp partitioning. The FA1 paper itself
describes the tiled algorithm but does not describe this warp-level mapping.

#### i) FA1 warp partitioning:

The optimized v1 uses the "split-K" approach for matrix multiplication. It uses
warps to divide the key-sequence dimension B_c, which is the common/reduction
dimension of P @ V, and later reduces the warp-local partial outputs into the
block-level result.

In FA1's optimized kernel, a warp owns the output produced using the entire
Q tile while only using a key-sequence partition of the current K/V data tile.
Since queries represent the rows of the score and K/V positions represent its
columns, the score output of a warp is [B_r, C], where B_c/#warps = C (I would
use K here if it didn't represent key).

Note: my implementation of FA1 was simpler and more serial than this.

Ex:

```text
S_iw = Q_i @ K^T_w          # warp-local score-column slice
S_i = [S_i1 S_i2 ... S_iw]  # logical concatenation along the col dim
```

The problem is that calculating the softmax of the score requires communication
between warps because an entire score row is needed, but each warp only owns a
column subset of each row.

When later multiplying by V, a similar issue occurs.

#### ii) FA2 warp partitioning:

V2 splits the Q data tile such that each warp calculates the result for only a
row slice of the Q tile while using all of the current K^T and V tile.

Ex:

```text
S_wi = Q_w @ K^T_i
```

Since each warp owns full Q rows, that means it also fully owns the corresponding
rows of the score and attention output. If the warp results were logically combined,
they would stack by row.

```text
S_i = [
    S_0i
    ...
    S_wi
]
```

Because attention is independent among score rows, the score and other state such
as m_i, l_i, and the output are all owned by the warp! So global memory
or shared memory copies are not needed and synchronization isn't needed
except for retrieving Q, K and V. Note: shared memory may still be used in the
implementation, but it is no longer required by the algorithm.

---

## Algorithm:

Due to independent query-row ownership between warps, shared memory for
intermediate softmax state and output accumulators does not need to be used as
heavily as with my FA1 implementation. Shared memory may still be used to stage
Q/K/V tiles or rearrange matrix fragments.
Normal CUDA can be used for element-wise operations, but optimized matrix
multiplication should use lower-level primitives, and warp-local reductions can
use warp shuffling.

For the first implementation, shared memory can be used to make this easier.
Thus, inter-warp operations can be done in shared memory while the rest
will remain warp local if possible, including borrowing v1's warp reductions.

```text
Initialize:
    B_c = 32    # Number of score cols and K/V rows
                # processed per data tile (can be tuned).
    B_r = 64    # Number of score rows and Q rows
                # processed per data tile (can be tuned).

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks

    # This represents conceptual state for one Q tile. Each warp owns the
    # entries corresponding to one or more complete query rows.

    m_i = fill(B_r, -infinity) # (B_r): Running maximum state for a Q tile.
    l_i = zeros(B_r)           # (B_r): Running softmax denominator state.
    O_i = zeros(B_r, D)        # (B_r x D): Unnormalized running output state.



# Each thread block selects rows of queries and calculates rows of O output.

# -- Load shared memory 2D tile dim (B_r x D) --

s_Q_i = Q[b, h, i*B_r: min((i + 1)*B_r, M), :]


for each K/V block j = 0 to T_c - 1:

    s_K_j = K[b, h, j*B_c: min((j + 1)*B_c, N), :]
    s_V_j = V[b, h, j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Save up to B_c rows of K and V into SRAM.


    s_S_ij = s_Q_i @ s_K_j^T / sqrt(D)

        # (B_r, B_c). Each warp owns one or more complete rows of S_ij.
        # This can be in shared memory for this implementation.


    w_mnew_i = max(m_i, rowmax(S_ij))

        # (B_r). Each warp calculates the entries for its owned rows. m_i
        # contains the running maximum from the previous K/V-tile iteration.


    s_P_unscaled_ij = exp(s_S_ij - w_mnew_i)

        # (B_r x B_c) Calculate running probabilities using the updated max
        # though skip applying softmax denominator here. This can re-use
        # S_ij allocated sram.


    w_m_i_rescale_factor = exp(w_m_i - w_mnew_i)

        # (B_r): By multiplying by this constant, the exp scale of the previous
        iteration is updated to this iteration's max.

    w_l_i = m_i_rescale_factor*l_i + rowsum(P_unscaled_ij)

        # (B_r): Update running denominator. By the end of the algorithm,
        # l_i is the denominator for each row of rowsoftmax(S).

    # -- Compute running output --

    w_O_i = _O_i * m_i_rescale_factor + P_unscaled_ij @ V_j

        # (B_r x D): Rescale softmax numerator for running output then
        # add this tile to the running output.

    w_m_i = w_mnew_i

# Apply the softmax denominator once after processing every K/V tile.
O[b, h, i*B_r: min((i + 1)*B_r, M), :] = O_i / l_i
```

---

The following implementation is a simplified version of FA2. For ease of
implementation, it stores some state in shared memory even where FA2's
warp-local ownership would allow that state to remain in registers.

Variables are named similar to the FA1 paper: the s prefix means shared mem
ptr, the g prefix means global mem ptr, and _i means a subscript of i.
