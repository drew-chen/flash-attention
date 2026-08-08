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

V5 applies tensor cores to both block-local matrix products: `S = Q @ K^T` and
`O = P @ V`.

V5 uses WMMA for both products while preserving the FA2-style warp ownership
used by v3 and v4. The running O state stays in registers across K/V tiles. For
each P @ V tile, WMMA produces only the new contribution, which is briefly
materialized in per-warp shared memory because WMMA intentionally hides the
accumulator fragment's lane-to-element mapping. Each lane then adds its two
owned output columns to the register-resident O state. Q @ K^T uses the same
shared-memory round trip to convert its accumulator fragment into the score
columns owned by each lane.

### WMMA instruction mapping

V5 uses `m8n32k16`. For Q @ K^T, `[8,64] @ [64,32]` already matches the
instruction's `m=8` and `n=32`, while its shared dimension is `64`. Therefore,
four `k=16` WMMA operations accumulate into the same `[8,32]` output fragment.

P @ V computes `[8,32] @ [32,64] = [8,64]`. Because `m8n32k16` produces an
`[8,32]` output, the result is a `1x2` grid of WMMA output tiles. `O0` is the
left tile accumulated in `wO_i_left_unnormalized`, and `O1` is the right tile
accumulated in `wO_i_right_unnormalized`. Each spans all eight query rows and
32 of the 64 output columns, requiring two `k=16` WMMA operations to reduce
over the 32 K/V positions:

```text
                 shared dim (32 K/V positions)
P [8,32]       = [ P0 [8,16]  | P1 [8,16]  ]

                   output columns
V [32,64]      = [ V00 [16,32] | V01 [16,32] ]  keys 0..15
                 [ V10 [16,32] | V11 [16,32] ]  keys 16..31

O [8,64]       = [ O0 [8,32]   | O1 [8,32]   ]

O0 = P0 @ V00 + P1 @ V10
O1 = P0 @ V01 + P1 @ V11
```


## Shared-memory padding

WMMA requires half-precision leading dimensions to be multiples of eight and
float accumulator strides to be multiples of four. The logical widths already
meet those constraints, but padding changes the shared-memory bank mapping and
reduces conflicts during collective fragment loads and stores.

| Buffer | Logical row width | Physical row stride | Purpose |
| --- | ---: | ---: | --- |
| K/V | 64 half | 72 half | Preserve WMMA alignment and improve operand-load bank mapping |
| P | 32 half | 40 half | Improve WMMA operand-load bank mapping |
| Per-warp WMMA result | 32 float | 36 float | Improve the store and lane-reload bank mapping |

The scalar v3/v4 P @ V loop can broadcast one P value while each lane keeps its
output columns directly in registers. V5 instead loads P collectively and
materializes WMMA results because their lane mapping is opaque, making padded P
and result buffers useful. These strides change only physical shared-memory
addressing; the logical tiles and attention calculation are unchanged.
