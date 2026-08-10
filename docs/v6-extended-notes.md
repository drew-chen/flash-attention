# V6 extended notes - Cheaper scalar operations

V6 keeps V5's WMMA operations, shared-memory layout, FP32 accumulation, and
synchronous K/V staging. It changes only two scalar operations:

- `__expf` replaces `expf` in the online-softmax update.
- Five `__shfl_xor_sync` steps replace five down-shuffles plus a broadcast for
  warp reductions.

These changes preserve measured FP16 accuracy while reducing scalar work.

## Results

At `B=4`, `H=12`, `M=N=2048`, and `D=64`, 21 randomized interleaved samples of
100 calls measured:

| Metric | V5 | V6 | Change |
| --- | ---: | ---: | ---: |
| Benchmark latency | 1864.29 us | 1663.62 us | -10.8% |
| Profiled duration | 2.042 ms | 1.823 ms | -10.7% |
| Long Scoreboard | 0.954 | 1.049 | +10.0% |
| Short Scoreboard | 6.277 | 5.064 | -19.3% |
| Registers/thread | 64 | 64 | unchanged |
| Achieved occupancy | 48.31% | 48.31% | unchanged |

V5 and V6 had identical measured errors against the FP32 reference: max
absolute error `1.098e-4`, RMSE `1.025e-5`, and relative L2 `2.819e-4`.

## Experiments not retained

| Experiment | Result |
| --- | --- |
| Explicit QK/PV fragment ping-pong | No measurable change and still 64 registers/thread. NVCC likely already unrolls and schedules these fixed-trip loops effectively. |
| Deferred output scaling | Short Scoreboard fell from 5.064 to 4.55, but registers rose to 72 and latency regressed about 1.2%, V5-normalized. |
| Independent left/right PV accumulators | Halved P-fragment loads and reduced Short Scoreboard to 4.24, but used 72 registers, raised Math Pipe Throttle from 0.58 to 0.91, and regressed about 2.7%. |
| Asynchronous V staging | Reduced Long Scoreboard about 47%, but improved latency only about 0.4% and used 75 registers. |
| Asynchronous K staging | Had too little independent work to overlap effectively. |
| Four-block residency | Gained about 2% with substantially more shared-memory and register-lifetime complexity. |
| Retaining both P fragments | Gained less than 0.5%. |
| Hoisting the warp's P pointer | Hoisted and unhoisted versions generated identical SASS. |

Reducing a stall counter did not necessarily reduce runtime: existing resident
warps already hid some latency, while longer register lifetimes and shifted
execution-pipeline pressure introduced new costs. The simpler synchronous
64-register kernel was retained.
