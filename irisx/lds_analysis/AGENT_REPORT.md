# AGENT 10 REPORT — LDS-pressure / occupancy-ceiling investigator

## Build command (main agent, on <NODE>, gentle + locked)
```
docker exec r1_c4 bash -lc '
  cp -r <repo>/irisx/lds_analysis/v4_bsingle_buffer \
        <HK_ROOT>/distributed-kernels/v4_bsingle_buffer
  cp <HK_ROOT>/distributed-kernels/v4_astationary_kernel/example.py \
        <HK_ROOT>/distributed-kernels/v4_bsingle_buffer/example.py   # ABI identical to V4
  cd <HK_ROOT>/distributed-kernels
  flock /tmp/mi355x_compile.lock -c "
    cmake -B build_10 -DDK_BUILD=v4_bsingle_buffer ;
    cmake --build build_10 -j8 --target v4_bsingle_buffer"'
```
Static resource report (subagent-safe; confirms the LDS claim):
```
HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage -c kernel.cpp -I<HK_ROOT>/include
# expect fused micro_tk LDS ~82944 B  (vs ~148480 B canonical V4).
```

## Proposed GPU test command (MAIN AGENT ONLY — serialized)
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/v4_bsingle_buffer
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    M=1024 K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"'
```

## Expected output
- Fused µs at M1024/N2048/K7168 within noise of canonical V4 (~677 µs) — single-buffer-B should be
  perf-neutral at occ 1 (the removed B prefetch was cheap local overlap).
- Static LDS for fused micro_tk ~82944 B (down from 148480). rocprof blocks/CU should still read 1
  for NSUB=8; rebuild with `NSUB 6` define to see it reach 2.

## Correctness criteria
- RMS-rel vs reference ~0.0033 (unchanged from V4 — change is LDS layout, not math).
- zero-sentinel still proves the remote IRIS gather path.

## Key findings (full detail in LDS_ANALYSIS.md)
- Exact formula (from `HK/include/cdna4/types/shared/st.cuh:81` + `KITTENS_DEFAULT_ALIGN`):
  `LDS = NSTAGE*(BM*BK*2 + NSUB*BN*BK*2) + 1024`. Default = `2*(8192 + 8*8192) + 1024 = 148480 B
  ~145 KB`. At 160 KB/CU → 1 block/CU regardless of VGPR. Matches Agent 07.
- **Dominant term = the B buffers** (NSTAGE*NSUB*ST_B = 131072 B = 88%). A is only 11%, and the
  A-stationary win lives entirely in A — so cut B freely.
- **Single-buffering B is safe** (B is local HBM, no remote latency to hide; A stays double-buffered
  so the expensive cross-GPU gather still overlaps). Saves **65536 B (~64 KB)**; A-stationary win
  fully preserved (N_PER_BLOCK unchanged). Alone it gives occ 1 (81 KB, 160/82.9=1.93); pair with
  BK=32 (→3 blocks) or NSUB=6 (→2 blocks) to gain occupancy.

## Files written (all under irisx/lds_analysis/)
- `LDS_ANALYSIS.md` — byte budget, 6 levers, ranked recommendation, Agent-07 cross-check.
- `v4_bsingle_buffer/kernel.cpp` — asymmetric-buffering variant (A double, B single). The
  highest-value concrete artifact; compile + measure first.
- `v4_bsingle_buffer/README.md` — exact diff-from-V4 + build/test commands.
- `AGENT_REPORT.md` — this file.

## Assumptions
- LDS_CAP = 160 KB/CU (MI355X/CDNA4, 2× CDNA3). **[NEEDS-NODE]** via rocminfo/device props.
- `sizeof(st_bf<64,64>) = 8192 B` exact, no padding — verified from headers.
- Perf-neutrality of single-buffer-B and the occ-2/3 predictions are HYPOTHESES pending rocprof.

## Known risks
- If LDS_CAP is actually 64 KB, no NSTAGE=2 fused config fits; BK/NSUB must shrink (V4 reportedly
  runs → evidence cap ≥ 145 KB).
- Single-buffer-B serializes producer-write/consumer-read of B via the existing per-tile s_barrier;
  if the local B load turns out to be on the MFMA critical path, expect a small regression at occ 1
  (recovered once occupancy rises). Measure.
- NSUB=6 introduces N_PER_BLOCK=384 ∤ 2048 → tail subtiles need masking (Agent 07's note); verify
  correctness on N=2048 before trusting timings.

## Static resource info
- NOT compiled by this agent (no Bash/ssh access; design-only). LDS bytes are derived analytically
  from the HK headers; main agent must produce the `-Rpass-analysis` report to confirm.

## What the main agent must compile/measure
1. `v4_bsingle_buffer` NSUB=8 — confirm LDS ~82944 B, RMS-rel ~0.0033, µs vs V4 (expect ≈).
2. Rebuild with `-DNSUB=6` (or edit the define) — confirm LDS ~66560 B and rocprof blocks/CU = 2.
3. Rebuild with `-DBK=32` — confirm LDS ~41984 B and rocprof blocks/CU = 3; this is the top pick.
