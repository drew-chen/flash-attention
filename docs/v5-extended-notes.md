# V5 extended notes - What are tensor cores and how can they be used here?

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

V5 uses WMMA for both products while preserving the FA2-style warp ownership
used by v3 and v4. The running O state stays in registers across K/V tiles. For
each P @ V tile, WMMA produces only the new contribution, which is briefly
materialized in per-warp shared memory because WMMA intentionally hides the
accumulator fragment's lane-to-element mapping. Each lane then adds its two
owned output columns to the register-resident O state. Q @ K^T uses the same
shared-memory round trip to convert its accumulator fragment into the score
columns owned by each lane.

To use WMMA, we mainly need to think about the shape of each multiply. Using
W_M = 8, W_N = 32, W_K = 16, `m8n32k16` evenly divides Q @ K^T because
`[WARP_TILE_ROWS, D] @ [D, B_c] = [WARP_TILE_ROWS, B_c]`, or more concretely,
`[8, 64] @ [64, 32]`. P @ V uses two K=16 steps for each of two 32-column
output tiles. The half-precision leading dimensions must be multiples of eight.
K and V therefore use a padded shared-memory row stride of 72 rather than 66;
the padding also reduces shared-memory bank conflicts.

## Why V5 needs new shared-memory padding

V3, V4, and V5 use the same high-level 64-row query tiles, 32-row key/value
tiles, and eight warp-owned query rows. The new padding is not caused by a new
high-level tiling scheme. It is caused by replacing scalar lane-owned matrix
products with collective WMMA fragment loads and stores.

The scalar V3/V4 PV loop iterates over one key row at a time. Every lane reads
the same `P[row, key_row]`, which shared memory can broadcast, while the lanes
read consecutive V columns. P therefore does not need a padded row stride in
that access pattern. The scalar result mapping is also explicit: each lane owns
two output columns and keeps them in registers, so no shared result tile is
needed. V4 does pad K from 64 to 65 floats for its separate scalar QK access
pattern; the earlier kernels were not padding-free generally.

WMMA changes both cases. `load_matrix_sync` distributes an entire P fragment
across the warp through `LDSM` instructions instead of broadcasting one scalar
P value. V5 therefore keeps 32 logical P columns but uses a 40-half physical
row stride. WMMA also hides which accumulator elements belong to each lane, so
V5 uses `store_matrix_sync` to materialize each logical `[8,32]` result before
lanes reload their columns. That buffer uses a 36-float physical stride instead
of 32. K and V use a 72-half physical stride to satisfy WMMA alignment and
reduce their remaining `LDSM` conflicts.

These paddings change only physical shared-memory addressing. The logical
matrix shapes and attention calculation are unchanged.
