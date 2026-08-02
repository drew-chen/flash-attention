# V1 extended notes

These notes construct tiled FlashAttention step-by-step from standard
attention. The derivation of the recurrence relation is omitted.

Algorithms:

1. Standard attention
2. Single-row, two-pass online-softmax attention
3. Single-row, single-pass online-softmax attention
4. FlashAttention (tiled)

Algorithm 4 is the one that is implemented.

`flash_forward_v1_cuda_launch` expects raw pointers for tensors shaped as:

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

Indexing convention: tensor indices are zero-based. A loop over N elements uses
i = 0 to N - 1. Recurrence state 0 is the empty-prefix state, so processing
tensor element i advances recurrence state i to state i + 1.

## 1. Standard attention algorithm:

```text
S = QK^T / sqrt(D) (pre-softmax logits, i.e., scores)
P = row_softmax(S) (attention probabilities/weight matrix)
O = PV  (attention output)
```


## 2. Single-row, two-pass online-softmax attention:

Algorithm for one row of output O[b, h, k, :], with b, h, and query row k fixed.
This algorithm avoids materializing the full attention matrix, but saves one score row x.

Notes taken from Zihao Ye's "From Online Softmax to FlashAttention".

Derivation of online softmax's recurrence relation is not shown.


### Pass 1:

```text
Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.

for i = 0 to N - 1:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)  # Scalar logit for query k and the current key i.
    m_{i+1} = max(m_i, x_i)                # Running maximum for the score row.
    l_{i+1} = l_i*e^(m_i - m_{i+1})        # Update the running max-shifted softmax normalizer.
            + e^(x_i - m_{i+1})
save x_i values for this score row
```

### Pass 2:

```text
Initialize:
    o_0 = zeros(D)  # Running partial attention-output row vector.

for i = 0 to N - 1:
    a_i = e^(x_i - m_N)/l_N         # Calculate the numerically stable attention weight.

    o_{i+1} = o_i + a_i*V[i, :]     # Accumulate the weighted value row.
                                    # Over N iterations, this is equivalent to row vector
                                    # a * matrix V, since the full attention-weight row
                                    # a is dotted with each column of V;
                                    # equivalently, each row V[i, :] is scaled by a_i
                                    # before the rows are summed.

    O[b, h, k, :] = o_N             # Save row vector output
```


## 3. Single-row, single-pass online-softmax attention

Using a flash attention recurrence relationship yields:

```text
Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.
    o_0 = zeros(D)   # Running normalized attention-output row vector.

for i = 0 to N - 1:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)           # Scalar logit for query k for each key vector.
    m_{i+1} = max(m_i, x_i)                         # Update the running maximum.
    rescaled_l_i = l_i*e^(m_i - m_{i+1})            # If prev max was the same, do nothing,
                                                    # otherwise, correct its exponent scale.

    l_{i+1} = rescaled_l_i + e^(x_i - m_{i+1})      # Update attention row's running
                                                    # sum's softmax denominator.

    old_output_contribution = o_i * l_i * e^(m_i - m_{i+1}) / l_{i+1}

        # Remove prev output's denominator l_i
        # then correct its exponent scale and set the newly updated denominator l_{i+1}.

    o_{i+1} = old_output_contribution + (e^(x_i - m_{i+1})/l_{i+1})*V[i, :]

        # Add this row vector to the running sum output row vector

O[b, h, k, :] = o_N                 # Save row vector output
```

## 4. FlashAttention (tiled)

Unlike the previous examples, this is for the entire output rather than a row.
Furthermore, the notation is adjusted from the paper to more closely align with
CUDA. Furthermore, the subscript annotation is for indexing into block-level
state rather than for state transitions like above.

Divide Q, K, and V along the sequence dimension and load into shared memory 2D tiles
of (B_r x D), (B_c x D), and (B_c x D) respectively, where (B_r, B_c) are the
dimensions of the score tile calculated by processing the queries, keys, and values
in the tile.

Ex: Q (M x D) is composed of T_r tiles, labelled Q_i (B_r x D) by stacking along the sequence dim.

```text
Q = [
    ---Q_0---
    ---Q_1---
    ...
    --- Q_{T_r - 1}
]
```

Through algebra similar to how online-softmax is performed, attention can be performed
one tile at a time with a running output. Only the current Q and K/V tiles and running
state slices need to be simultaneously loaded into shared memory.


This avoids operating on all M or N elements at once and eliminates the need to
materialize the M x N attention score matrix (for self-attention, M = N).


### Intuition on running state:

After iterating over the entire sequence and all blocks, the running output is the same
as the naive attention output. The running state m_i, l_i, and O_i lives in global
memory between tile updates. The current tiles are loaded into shared memory, where m_i and l_i
have one scalar per query and O_i has one D-element row per query.

The m_i and l_i state must be stored as vectors, not single variables, because we do not
actually calculate the result for one query's attention before moving onto the next; we
incrementally build running state for a block of queries.

The outer loop iterates over K/V tiles, thus for a fixed K/V tile, we iterate over all
query tiles to calculate the running output. This is an inversion of the
per-query output perspective but is mathematically equivalent and keeps the heavier
data movement of the K/V tiles on the outer loop rather than inner loop.


```text
Initialize:
    B_c = floor(shared_memory_capacity_elements/(4*D)) # Number of score cols and K/V rows
                                                # processed per data tile.
    B_r = min(B_c, D)                           # Number of score rows and Q rows
                                                # processed per data tile.

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks

    # Allocate a shared-memory workspace for the current Q, K/V, score, output,
    # and recurrence-state tiles. Allocate global-memory backing for m and l.
    # O serves as both the output and the backing for the running output state.

    m_i = fill(B_r, -infinity) # (B_r): Shared running maximum state for a Q tile.
    l_i = zeros(B_r)           # (B_r): Shared running softmax denominator state.
    O_i = zeros(B_r, D)        # (B_r x D): Shared running attention output state.


# Keep K/V as the outer loop so each loaded K/V tile is reused across all query tiles.

# The outer loop uses j to stay consistent with the FlashAttention paper.

for each K/V block j = 0 to T_c - 1:

    K_j = K[j*B_c: min((j + 1)*B_c, N), :]
    V_j = V[j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Save up to B_c rows of K and V into shared memory.

    for each Q block i = 0 to T_r - 1:
        # -- Load shared memory 2D tile dim (B_r x D) --
        Q_i = Q[i*B_r: min((i + 1)*B_r, M), :]

            # (B_r x D): Choose up to B_r rows of Q.

        # Load this Q tile's current m_i, l_i, and O_i into shared memory, using
        # the initialized defaults for its first update and globally saved state later.


        # -- Calculate the block-local scores using shared-memory tiles --

        S_ij = Q_i @ K_j^T / sqrt(D)

            # (B_r x B_c): S_ij is a score matrix, calculating the score for each Q
            # row in the block by dotting with each K row in the block.
            # A row of S_ij is only the score of a query against B_c rows of
            # K rather than all N key rows.
            # Note 1: The transpose is conceptual and done by changing indexing,
            #   rather than a separate device function call.
            # Note 2: The rest of the work here is building a running output
            #   from the segment S_ij which eventually yields the same result
            #   using S.

        # -- Setup block-local variables for block-local softmax --

        # (this is in parallel for each query in the block).

        mlocal_ij = rowmax(S_ij)

            # (B_r x 1): block local rowmax to eventually calculate the running max for a query

        mnew_i = max(m_i, mlocal_ij)

            # (B_r): Update the vector so each element corresponds to
            # the running max score for each query in the block
            # By the end of the entire algorithm, m_i will contain
            # rowmax(S), which is equivalent to what
            # would've happened had we fully materialized S and performed
            # rowsoftmax.

        rescaled_l_i = l_i*e^(m_i - mnew_i)

            # (B_r): This represents the old block local rowsoftmax's denominator
            # prior to the contribution of the scores using this query block.
            # Conceptually, we are updating a query's row-softmax denominator
            # by using the scores of new K/V rows. This is confusing because
            # we have the inner loop iterate over queries for memory performance,
            # while I phrase these concepts from a fixed query perspective.
            # If prev max was the same, do nothing, otherwise, correct its exponent scale
            # to use the updated max.

        new_l_i_contribution = rowsum(e^(S_ij - mnew_i))

            # (B_r): Apply exp operations by broadcasting, subtract each row of S_ij by its
            # corresponding row maximum, then calculate the row sum.
            # This is this tile's contribution to the softmax denominator for
            # each query.

        lnew_i = rescaled_l_i + new_l_i_contribution

            # (B_r): Update running denominator. By the end of the algorithm,
            # l_i is the denominator for each row of rowsoftmax(S).

        # -- Calculate this query block's contribution to the running output --

        P_ij = exp(S_ij - mnew_i) / lnew_i

            # (B_r x B_c) Calculate softmax row-wise to scale tile scores into probabilities
            # using mnew_i and lnew_i, broadcast across their corresponding rows.
            # For each row of S_ij, subtract the corresponding row max, then divide
            # by corresponding denominator. Each row of S_ij again represents
            # the score from one query when scored against the K/V rows in this tile.
            # Output is a 2D matrix the same size as S_ij.

        new_output_contribution = P_ij  @  V_j

            # (B_r x D): Conceptually, this does this:
            # For every row of P_ij, i.e., for every query in the sequence tile,
            #   dot product with every value in the tile and sum them,
            #   to effectively perform a weighted sum of probabilities.
            # Finally, by parallelizing across B_c rows of V_j, we finish the calc
            # for this tile.
            # This basically means we attend each query with every key and value
            # within our tile.


        # -- Adjust running output for this query with the updated running max --

        rescaled_O_i = O_i * rescaled_l_i / lnew_i

            # (B_r x D): For the running O_i matrix, update the scalars' safe
            # softmax factors via scaling so that they are re-calculated with mnew_i.
            # Do this by multiplying by the rescaled old denominator contribution,
            # then dividing by the new denominator lnew_i.


        # -- Save running output for this query --

        Onew_i = rescaled_O_i + new_output_contribution

            # (B_r x D): Sum matrices to update tile's running output

        # -- Save running state to global memory --

        m[b, h, i*B_r: min((i + 1)*B_r, M)] = mnew_i
        l[b, h, i*B_r: min((i + 1)*B_r, M)] = lnew_i
        O[b, h, i*B_r: min((i + 1)*B_r, M), :] = Onew_i

            # Save the updated state. O is finalized after the last K/V tile.

        # A future optimization is to store the O tile without the li denominator
        # and perform the division on a second pass on the final O tile, reducing # FLOPs
        # otherwise needed to rescale constantly.
```

---

In the following impl, variables are named similar to the FA1 paper
and s prefix means shared mem ptr, g prefix means global mem ptr,
and finally _i means a subscript of i.
