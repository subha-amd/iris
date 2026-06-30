# Unfused vs Fused Baseline Findings

Date: 2026-06-30 UTC

This note records what we learned while trying to set up an accurate DeepSeek-R1
unfused-production baseline for comparing the IRISX MoE kernels. The goal is not
just to beat a standalone kernel. The goal is to know whether replacing the
production MoE path with our kernels would reduce real vLLM/ATOM decode latency.

## Desired comparison

The comparison we need is:

```
current production MoE path in vLLM/ATOM
vs
the same production path with only the MoE dispatch/GEMM/combine implementation replaced
```

For a production speedup claim, both sides must use the same:

- model checkpoint: DeepSeek-R1-0528
- topology: TP=4, DP=2, using all 8 GPUs as two DP replicas of four TP ranks unless ATOM config says otherwise
- routing and top-k policy
- batch/concurrency and prompt/output lengths
- graph mode / scheduling mode
- AITER/MORI backend configuration
- low-latency pathway disabled
- warmup count and timing window
- rank synchronization and max-over-ranks reduction

The headline number should be a per-decode-step MoE-region latency in nanoseconds
or microseconds, reduced with `MAX` across all participating ranks. End-to-end
TPOT/ITL from `vllm bench serve` should be collected too, but the MoE-region number
is the cleanest denominator for kernel work.

## What the current 1.76x result means

The ledger's 1.76x result is useful, but it is not yet the full drop-in
production comparison.

The current comparison is:

| Path | Scope | Time |
|---|---|---:|
| `b2_production/b2_aiter.py` | single-GPU `aiter.fused_moe` harness on already-local tokens | 1255 us |
| `b1_dispatch` with `SCHEDULE=b0` | consumer-rank gather/pack once plus local grouped GEMM | 714 us |

This is an apples-to-apples local-compute style comparison in the sense that both
measure work on a consumer GPU and neither measures the full distributed vLLM step.
It proves the b1_dispatch dataflow can beat the local production AITER harness for
that synthetic route and shape.

It does not prove a full production drop-in win because the denominator does not
include the real TP=4/DP=2 vLLM/ATOM MoE region with MORI dispatch/combine,
rank-to-rank synchronization, graph scheduling, and the real per-layer/per-step
expert distributions.

## What the profiling repo currently contains

Repository cloned locally:

```
/Users/subha/repos/public-kernels-testing
commit 1505a8fdb1972dee8874d23e57c17c2028808974
```

It was already up to date when checked.

Important finding: the checked-in scripts describe a TP=8 expert-parallel R1 setup,
not a TP=4 DP=2 setup.

Relevant examples:

- `scripts/profile_offline_generate.py` has `--tp`, but no `--dp`.
- `scripts/profile_serve.sbatch` launches `vllm serve` with `--tensor-parallel-size 8`
  and `--enable-expert-parallel`.
- The documented model path is `/shared/subvadla/models/DeepSeek-R1-0528`.
- The documented production-style image is `rocm/atom-dev:vllm-v0.22.0-nightly_20260610`.
- The repo emphasizes AITER plus MORI/RCCL, `ROCM_AITER_MLA`, graph mode, and expert parallelism.

So this repo is useful for profiling mechanics and prior TP=8 results, but it does
not currently provide the exact TP=4 DP=2 recipe requested.

## Remote node findings

Node used:

```
ssh -i ~/.ssh/muhammad-gpu -p 2424 subvadla@172.19.164.255
hostname: cv350-rck-g03-f03-18
GPU: 8x AMD Instinct MI350X / gfx950
```

The GPUs were idle, but a fresh DeepSeek-R1 TP=4 DP=2 run could not be launched
honestly on this node during this check because:

- `/shared` is not mounted on this node.
- `/shared/subvadla/models/DeepSeek-R1-0528` is absent.
- `/models/deepseek-ai/DeepSeek-R1-0528` is absent.
- Only about 117 GB was free on `/`, far below what is needed for the full R1 checkpoint.
- The running Docker container is `qilihuan-dsv4-dp8-ep-vllm0617` using image
  `sabreshao/vllm:dsv4_0615n`; it is mounted for DeepSeek-V4-Pro, not R1.
- The exact image named in the profiling repo,
  `rocm/atom-dev:vllm-v0.22.0-nightly_20260610`, was not present in the image list checked.

There are existing R1 artifacts under `/tmp/atom-ci-bench`, including benchmark
JSONs and all-rank Perfetto traces. Those artifacts are valuable for attribution
and kernel-name discovery, but they are not the requested TP=4 DP=2 baseline.

Existing JSONs found there showed TP=8 DP=1 standard R1 runs and TP=4 DP=1 MXFP4
R1-style runs. I did not find an existing TP=4 DP=2 R1 result row.

An all-rank trace parser was briefly started for:

```
/tmp/atom-ci-bench/regression-traces-27069055455/deepseek-r1-0528-1024-1024-32
```

The intended output target was:

```
/tmp/atom-ci-bench/codex_r1_existing_tp8dp1_trace_summary.txt
```

It was interrupted because it was no longer needed for this note, and no completed
summary file was produced. If this parser is run later, the result should be
treated as TP=8 DP=1 historical evidence only, not as the drop-in TP=4 DP=2
denominator.

## Why rank-0-only Perfetto is not enough

A rank-0 trace can answer local questions:

- what kernels ran on rank 0
- approximate rank-0 region durations, if annotations exist
- whether kernels appear serialized or overlapped on rank 0

It cannot by itself prove the production MoE denominator because production latency
is determined by the slowest participating rank or DP group. For TP=4 DP=2 we need
all ranks, or at minimum a logged all-reduce `MAX` duration emitted by the program.

Rank 0 also does not necessarily see the full expert-token distribution. `M_e` must
be logged per layer, per step, and per rank or reconstructed from routing metadata
that covers the full topology.

## Accurate baseline requirement

The unfused baseline must measure the actual production MoE region. At minimum the
region should include:

1. routing/top-k output as consumed by MoE dispatch
2. MORI/ATOM EpDispatch or equivalent all-to-all token movement
3. production sorting/reordering
4. production dynamic quantization, if it runs in that path
5. AITER/CK fused MoE expert compute
6. production combine/reduction within top-k
7. MORI/ATOM EpCombine or equivalent return path
8. required synchronization or graph dependencies around the region

The fused candidate must be measured over the matching boundary. If our path skips
sorting by producing expert-major packed order directly, that is valid only if the
measured fused region also includes all work needed to make the production output
layout match what the next vLLM stage expects.

## Required instrumentation

Add timing at the MoE call boundary in the actual vLLM/ATOM path:

- HIP/CUDA events around the full MoE region on each rank
- `torch.distributed.all_reduce(MAX)` or equivalent to log the max region time
- optional sub-event timings for dispatch, local reorder/quant, fmoe, combine
- ROCTX ranges around the same regions for Perfetto verification
- per-layer/per-step route stats: `M_e` per expert, top-k, total routed rows, padding rows
- kernel names and dimensions printed or dumped to CSV

Collect all-rank Perfetto traces for a short measured window. The event timers give
the stable denominator; Perfetto verifies the boundaries, overlap, streams, and
kernel names.

## Data to save for reproducibility

For every baseline run, save:

- exact launch command
- full Docker image ID, not just tag
- vLLM, ATOM, AITER, MORI, PyTorch, ROCm versions and commits
- model path and checkpoint revision
- all relevant environment variables
- `rocm-smi --showtopo`, GPU model, driver version
- `vllm bench serve` JSON outputs
- per-rank MoE timing CSV with max-over-ranks already computed
- per-rank `M_e` CSV
- per-rank Perfetto traces
- kernel-name table with grid/block dimensions when available
- indication that low-latency pathway was disabled

## Benchmark matrix

Minimum production run matrix:

| Case | Purpose |
|---|---|
| TP=4 DP=2, graph mode, default MORI, low-latency off | target production baseline |
| Same run with all-rank Perfetto enabled for a short window | timeline validation |
| Same shape replay in standalone b1_dispatch | isolates kernel/dataflow capability |
| Integrated fused path, same boundary, same workload | actual drop-in speedup test |

The standalone replay is not sufficient for a production claim, but it is still
useful for debugging and for matching captured `M_e` distributions before doing the
hard integrated run.

## Current conclusion

The b1_dispatch `SCHEDULE=b0` result is promising and the ledger's 714 us vs 1255 us
comparison is a valid local harness comparison. It should not be presented as a
complete production drop-in speedup yet.

To make the benchmark accurate, the next necessary step is to run or obtain the exact
DeepSeek-R1 TP=4 DP=2 production vLLM/ATOM workload with default MORI, low-latency
disabled, and MoE-region event timing plus all-rank Perfetto traces. Without that,
we do not have the denominator our fused kernels must beat.
