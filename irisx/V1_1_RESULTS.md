# V1.1 MoE dispatch+pack+FP8-quant (PIPELINED) — results (8x MI355X, gfx950, 2026-06-25)

V1.1 = V1 + two optimizations aimed at V1's QUANT bound (V1_RESULTS.md validation #5:
halving the payload only gave ~23%, not ~50%, because the bf16->fp8 conversion math costs
more than the remote stores, so the interconnect sits partly idle).

V1.1 keeps everything else identical to V1: output layout, count/offset machinery,
sender-owned slice (no remote atomics), bulk completion signal, device-queried grid
(256 CUs), 256-thread block = 8 independent 32-lane tiles, PROFILE mode.

## The two optimizations
1. **OPT1 — software-pipeline quant(g+1) over the fire-and-forget store(g).**
   V1 ran `pack(g) -> store(g) -> pack(g+1) -> store(g+1) ...` sequentially per tile.
   `iris::view.store` is `*remote_ptr = value` (non-blocking, no wait), so V1.1 holds the
   produced 32-bit word + fp32 scale of group g in registers, ISSUES its store, then runs
   the quant math of group g+1 while that store is in flight (prologue / steady-state /
   epilogue). Goal: total time -> max(quant, store) instead of sum(quant, store).
2. **OPT2 — packed fp8x2 convert.** Replaced the 4x `__hip_cvt_float_to_fp8` per lane with
   2x `__hip_cvt_float2_to_fp8x2` (each emits a 16-bit fp8x2, OR'd into the 32-bit word).
   **CAVEAT (fall-back path, documented):** in this ROCm 7.2.4 `hip/hip_fp8.h`,
   `__hip_cvt_float2_to_fp8x2` is NOT a hardware packed instruction — it is a SOFTWARE
   wrapper that internally calls `__hip_cvt_float_to_fp8` twice
   (`amd_hip_fp8.h:743-745`). So OPT2 expresses the 2-wide intent but cannot cut the
   convert cycles on this stack; there is no true fp8x2 op to fall back from. This is the
   honest reason OPT2 contributes ~0 here (see attribution).

## Build (verified, gfx950)
```
source /usr/share/Modules/init/bash && module load mpi/openmpi-x86_64
cd <repo>/irisx   # node path: see ../NODE_ACCESS.local.md
cmake -B build -DIRIS_BUILD_BENCHMARKS=ON -DIRIS_BUILD_TESTS=ON -DIRIS_HIP_ARCHITECTURES=gfx950
cmake --build build --parallel 8 --target moe_dispatch_pack_quant_pipelined test_moe_dispatch_pack_quant_pipelined
```
Builds clean for gfx950 (only the same benign `hipMemsetAsync` nodiscard warning as V1).

## Run (force shared-mem MPI transport; node IB can't register memory)
```
mpirun --mca pml ob1 --mca btl self,vader -np 8 ./build/tests/test_moe_dispatch_pack_quant_pipelined
mpirun --mca pml ob1 --mca btl self,vader -np 8 ./build/benchmarks/moe_dispatch_pack_quant_pipelined      # add "1" for in-kernel profile
```

## Correctness — PASS, numerically identical to V1
`test_moe_dispatch_pack_quant_pipelined`: ALL TESTS PASSED on 8 ranks (9 assertions/rank).
```
[V1.1] checked 496 assignments, 0 multiset errors, 0 bad route_slot
[V1.1] FP8 e4m3 dequant vs bf16: max_rel_err=0.0476  mean_rel_err=0.01988  (n=55,776)
```
**Identical to V1** (V1 was max_rel_err=0.0476, mean=0.01988). Confirms the pipelining
reordering and the fp8x2 packing do NOT change numerical results — same per-128-group
scale, same per-element e4m3 quant. route_slot valid; per-(src,expert) multiset matches.

## Performance — before/after, T_local sweep (H=7168, topk=8, 256 experts, world=8; grid=256 CUs)
Real timing (PROFILE=0; the PROFILE path is heavily inflated by lane-0 clock64/atomicAdd
serialization and must not be read as wall time). V1 and V1.1 run back-to-back, same node.

| T_local | V1 us/inst | V1.1 us/inst | V1.1 vs V1 | V1 GB/s | V1.1 GB/s |
|--------:|-----------:|-------------:|-----------:|--------:|----------:|
| 8       | 16.7 | 16.5 | -1.2% | 28.2  | 28.7  |
| 16      | 19.0 | 19.0 |  0.0% | 49.9  | 49.7  |
| 32      | 22.5 | 23.0 | +2.2% | 84.1  | 82.2  |
| 64      | 32.1 | 31.7 | -1.2% | 117.9 | 119.4 |
| 128     | **45.8** | **44.8** | **-2.2%** | 165.3 | 169.1 |

Net: V1.1 is a **statistical tie with V1** (~2% at T=128, inside run-to-run noise). The
two optimizations did not move the end-to-end wall time meaningfully.

## Profile cycle split — what actually changed (PROFILE=1, directional not absolute)
| T_local | V1 quant-cyc | V1 store-cyc | V1.1 quant-cyc | V1.1 store-cyc | store-cyc change |
|--------:|-------------:|-------------:|---------------:|---------------:|-----------------:|
| 8       | 665.2M | 32.7M | 673.1M | 11.2M | **-66%** |
| 32      | 10.644B | 130.3M | 10.687B | 44.5M | **-66%** |
| 64      | 22.168B | 260.8M | 22.252B | 88.3M | **-66%** |
| 128     | 44.966B | 522.1M | 45.239B | 175.5M | **-66%** |

Two clear readings:
- **OPT1 (pipeline) worked at the micro level: remote-store cycles dropped ~3x (-66%)** —
  the store of group g now overlaps the quant of group g+1, so its measured cost shrinks
  toward the issue latency. But store was already only ~1% of quant cycles in V1, so
  hiding it under quant buys almost nothing end-to-end. The bottleneck never depended on
  the store.
- **OPT2 (fp8x2) did NOT reduce quant cycles** (45.0B -> 45.2B, unchanged/slightly up).
  Expected, given the intrinsic is a software wrapper over two scalar converts on this
  ROCm — same instruction count.

## Honest attribution — which optimization helped how much
- **OPT1 pipelining: correct, but near-zero payoff here.** It did exactly what it set out
  to do (store cycles -66%, store now fully hidden under quant), but because V1 was
  ~99% quant / ~1% store, removing the store from the critical path is in the noise.
  Pipelining is the right lever ONLY once the store is a meaningful fraction of the time.
- **OPT2 fp8x2: zero payoff on this stack.** No hardware packed fp8 convert exists in this
  ROCm `hip_fp8.h`; the "x2" call compiles to the same two scalar converts. To actually
  cut quant cycles we'd need a genuine HW packed convert or an inline-asm `v_cvt_pk_fp8`
  on gfx950, or to move the |max| reduction / convert work off the per-element path.

## Still quant-bound — bottleneck did NOT move
V1.1 is **still quant-bound, not byte-bound.** The store is now hidden (OPT1), which would
have helped a byte-bound kernel — but this kernel's cost is dominated by the on-chip quant
(load 4 bf16 + |max| sub-warp reduce + e4m3 convert), and neither optimization shrank that.
Confirmed by: store cycles ~0.4% of quant cycles in V1.1, and wall time unchanged.

## Recommended next step
The remaining win is purely in the quant math, since the store is already free:
1. **Real packed convert.** Use a true gfx950 packed fp8 convert (inline asm
   `v_cvt_pk_fp8_f32` / `v_cvt_pk_fp8_f16` if exposed) instead of the software wrapper —
   this is the one thing that would cut quant cycles. Verify the op exists on gfx950 first.
2. **Cheaper |max|.** The per-element `fabsf`+`fmaxf` then 5-step `__shfl_xor` reduce is
   on the critical path; try `v_max3`/packed-abs or a 2-wide load-and-reduce.
3. If quant truly cannot be cheapened, accept that this kernel is convert-throughput-bound
   and the ~46 us (T=128) single-kernel match of the production 3-kernel pre-GEMM envelope
   is the floor for this approach; the next lever is fusing into the expert GEMM (V2).

## Files (under irisx/)
- benchmarks/moe_dispatch_pack_quant_pipelined.hip   (V1.1 kernel: OPT1 pipeline + OPT2 fp8x2 + count + signal + PROFILE + sweep)
- tests/test_moe_dispatch_pack_quant_pipelined.hip   (Catch2, 8-rank: dequant vs bf16 + multiset + route_slot)
- benchmarks/CMakeLists.txt, tests/CMakeLists.txt    (registered moe_dispatch_pack_quant_pipelined / test_*)
- V1_1_RESULTS.md   (this file)
V1's files (moe_dispatch_pack_quant.hip / .../pack.hip and their tests) were left UNCHANGED
for a clean A/B.
