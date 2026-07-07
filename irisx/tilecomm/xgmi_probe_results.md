# On-node XGMI probe — results & honest interpretation (2026-07-07, 8× MI350X)

Ran `xgmi_probe.py` on `cv350-rck-g03-f03-18` (container `subha_tilecomm_probe`, IRIS pip-installed).
Fixed the earlier GPU memory-fault: `iris.store` cannot take a data-dependent, memory-loaded
`to_rank` (faults with "write access to read-only page"); the destination must be selected through
a constexpr `tl.static_range(WORLD)` loop, matching IRIS example 07. After the fix it runs clean.

## Raw numbers (µs, rank-0 timing, iris.do_bench)

| config | sorted | round_robin | proportional | sorted/rr |
|---|---|---|---|---|
| world=2, skew=0, nC=8192   | 55.8 | 55.1 | 54.1 | 1.01 |
| world=8, skew=0, nC=8192   | 59.2 | 57.4 | 55.9 | 1.03 |
| world=8, skew=1, nC=8192   | 56.0 | 53.1 | 50.9 | 1.06 |
| world=8, skew=2, nC=8192   | 57.6 | 54.3 | 52.7 | 1.06 |
| world=8, skew=0, nC=131072 | 349.2 | 347.6 | 348.4 | 1.005 |

## Interpretation — the probe is INCONCLUSIVE (issue-bound), not a refutation

The schedule barely matters here (sorted ≈ round-robin ≈ proportional, ~1.0–1.06×), which at first
looks like it *contradicts* the 2.4× combine win. It does not — the probe never reached the
bandwidth-bound regime:

- **Implied throughput is impossible for XGMI.** world=8 nC=131072 writes ~1.6 GB remote per rank in
  349 µs ⇒ **~4.7 TB/s**. The same node's `01_store` control measured **47 GiB/s** per link
  (~0.35 TB/s aggregate across 7 links). 4.7 TB/s is >10× the fabric — so the stores are **not
  completing inside the timed region.**
- **Root cause:** IRIS `store` is fire-and-forget; `do_bench` times kernel issue/return, and the
  async remote writes drain *after* the kernel returns. So the probe measures **issue rate, not
  link bandwidth** — and link *contention* only manifests on completion. That's why order looks free.
- The `01_store` control is completion-bound (it pushes 4 GB down one link, so issue can't outrun the
  link) — which is why it reports a believable 47 GiB/s. Our probe's 14 KB × N cells is too little per
  link to back-pressure the issue queue.

## Verdict

The on-node run **confirms the environment and the corrected `iris.store` usage**, and gives a real
control-plane number (47 GiB/s/link). It does **not** yet validate or refute the 2.4× combine result,
because the microbenchmark as written is issue-bound. The 2.4× stays a real *region-level* measurement
whose mechanism (link-spreading vs read/coalescing/reduction structure) is **not yet isolated**.

## Fix for a valid next run

Make the timed region completion-bound: (a) fence remote writes before stopping the timer (a
store-completion barrier / read-back of a sentinel per destination), and/or (b) push enough bytes
per link to back-pressure the issue queue (≥ a few GB per link, like `01_store`), and (c) time
MAX-over-ranks, not rank 0. Then re-run sorted vs round-robin at world=8.
