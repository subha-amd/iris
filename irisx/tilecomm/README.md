# tilecomm — tile-level communication abstraction (design + scheduler prototype)

The next research direction after the fused MoE region: turn the **hand-rolled communication
schedules** inside `irisx/fused_moe/` into a first-class, demand-aware library concern. Seed
for the paper *"Tile-Level Communication for Fused Multi-GPU Inference."*

**Motivation in one line:** the combine collective got **2.4× faster** (934 → 386 µs) from a single
hand-rolled schedule (`build_combine_pull(..., interleave=True)` round-robining cells across XGMI
links). That decision is workload-blind, non-reusable, and opt-in. TileComm makes the schedule
something the library computes from declared demand + measured link topology.

## Files

| file | what it is |
|---|---|
| `DESIGN.md` | the abstraction spec + paper framing: declare / schedule / execute, the 4 intents (`tile_gather/scatter/reduce_scatter/all_reduce`), the quadrant vs NCCL/MSCCL/aiter/IRIS/HK, how it maps onto our existing kernels, novelty, roadmap, honest limits. **Start here.** |
| `tilesched.py` | runnable scheduler + wave-based link-contention **cost model**, calibrated to the measured combine point (386/934). Three schedulers (`sorted`, `round_robin`, `proportional`) + a skew sweep. Pure numpy — runs anywhere. |
| `xgmi_probe.py` | on-node **validation** microbenchmark: measures sorted vs round-robin vs proportional store schedules directly on 8× MI350X via IRIS primitives (no full `fused_moe` build). A 2nd measured point for the cost model. |

## Run

```bash
python3 tilesched.py --json results.json        # the cost model (laptop-ok)
# on the 8x MI350X node (needs iris installed in a rocm+triton container):
python3 xgmi_probe.py --num_cells 8192 --H 7168 --world 8 --skew 0.0
```

## Key results (from `tilesched.py`, calibrated to real combine numbers)

- **Uniform combine** reproduces the measurement: sorted 927.8 µs / round-robin 388.2 / proportional
  388.3 (measured 934 / 386). The scheduling decision is worth **2.4×** over the naive order.
- **Imbalanced gather** (Zipf expert popularity): `proportional` tracks the link lower bound at every
  skew; `round_robin` drifts 1–4% and grows with imbalance; `sorted` is 4–5× worse.
- **Honest read:** reorder-only scheduling buys **2.4–6× over the naive order** (the big lever — now
  impossible to fall into by construction) but only **1–4% over the good hand-roll**. The unbounded win
  is Layer-3 **tile-fused comm/compute overlap** — future work the abstraction is built to express.

## Next step (concrete, low-risk)

Drop `tilecomm.schedule(transfer_set, topology)` in for the `interleave=True` argument in
`fused_moe/example.py`. Regression-gate at 386 µs on uniform traffic; measure the win on a real
load-imbalanced routing trace. Same `combine_pull_kernel`, measured against a number we already trust.

The presentation deck is in `../../july-07-presentation/` (self-contained HTML).
