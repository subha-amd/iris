#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# route_generators.py -- synthetic routing-buffer generators for the IRISX MoE ABI.
# CPU / numpy ONLY (no torch-GPU, no HIP). Emits exactly the buffers route_abi.h defines so
# the GEMM/dispatch agents can drive their kernels and the combine path without the GPU
# router. Distributions: uniform, Zipf, hot-expert, many-empty -- the four stress shapes that
# expose load imbalance, tail masking, and empty-expert handling.
#
# Output matches PRODUCTION_ABI.md 3 / ROUTE_CAPTURE_SCHEMA.md 3:
#   topk_ids[t,k]        int32   global expert id [0,256)
#   topk_weights[t,k]    float32 gate weight (rows sum ~1)
#   rows_per_expert[32]  int32   incoming rows per LOCAL expert (this rank)
#   expert_offsets[33]   int32   prefix sum
#   route_segment[]      one run per (expert, src_rank)
#   route_reverse[R]     packed-row -> (src_rank, src_token, topk_slot, weight)
#   route_params         self-describing header
# Plus a binary dump compatible with route_capture_file_header / route_capture_record.

import argparse
import json
import struct
import numpy as np

# ---- constants (mirror route_abi.h) ----
EP_SIZE = 8
N_GLOBAL_EXPERTS = 256
N_LOCAL_EXPERTS = 32
TOP_K = 8
HIDDEN = 7168
GROUP = 128
N_GROUPS = HIDDEN // GROUP  # 56
ROUTE_ABI_MAGIC = 0x52415831  # "RAX1"
ROUTE_ABI_VERSION = 1
SCALE_TOKEN_MAJOR = 0
SCALE_GROUP_MAJOR = 1


def _pick_topk(rng, probs, t_local):
    """Sample TOP_K distinct experts per token from a global-expert prob vector."""
    ids = np.empty((t_local, TOP_K), dtype=np.int32)
    for t in range(t_local):
        ids[t] = rng.choice(N_GLOBAL_EXPERTS, size=TOP_K, replace=False, p=probs)
    return ids


def _global_probs(kind, rng, hot_expert=0, n_empty=0):
    """Return a length-256 probability vector for the chosen distribution."""
    if kind == "uniform":
        p = np.ones(N_GLOBAL_EXPERTS, dtype=np.float64)
    elif kind == "zipf":
        ranks = np.arange(1, N_GLOBAL_EXPERTS + 1, dtype=np.float64)
        p = 1.0 / ranks  # Zipf s=1
        rng.shuffle(p)   # decouple expert id from rank
    elif kind == "hot":
        p = np.ones(N_GLOBAL_EXPERTS, dtype=np.float64)
        p[hot_expert % N_GLOBAL_EXPERTS] = N_GLOBAL_EXPERTS * 4.0  # one very hot expert
    elif kind == "many-empty":
        # only a small set of experts ever receive tokens -> most local experts empty
        p = np.zeros(N_GLOBAL_EXPERTS, dtype=np.float64)
        n_active = max(TOP_K, N_GLOBAL_EXPERTS - n_empty)
        active = rng.choice(N_GLOBAL_EXPERTS, size=n_active, replace=False)
        p[active] = 1.0
    else:
        raise ValueError(f"unknown kind {kind}")
    return p / p.sum()


def generate(kind, t_local, my_rank, seed=1234, hot_expert=0, n_empty=200,
             scale_layout=SCALE_GROUP_MAJOR, m_pad_to=32):
    """Generate the full ABI buffer set for one rank. Returns a dict."""
    rng = np.random.default_rng(seed + my_rank)
    probs = _global_probs(kind, rng, hot_expert, n_empty)
    topk_ids = _pick_topk(rng, probs, t_local)

    # gate weights: softmax over TOP_K random logits, rows sum to 1.
    logits = rng.standard_normal((t_local, TOP_K)).astype(np.float64)
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    topk_weights = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

    # this rank owns local experts [my_rank*32, my_rank*32+32). Keep only assignments that
    # land on THIS rank's experts (single-rank generator; src_rank == my_rank here).
    lo = my_rank * N_LOCAL_EXPERTS
    rows_per_expert = np.zeros(N_LOCAL_EXPERTS, dtype=np.int32)
    # collect (local_e, src_token, topk_slot, weight) for rows destined to this rank
    rows = []
    for t in range(t_local):
        for k in range(TOP_K):
            g = int(topk_ids[t, k])
            if lo <= g < lo + N_LOCAL_EXPERTS:
                le = g - lo
                rows_per_expert[le] += 1
                rows.append((le, my_rank, t, k, float(topk_weights[t, k])))

    # stable sort by local expert -> packed (expert-major) row order
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    total = len(rows)
    expert_offsets = np.zeros(N_LOCAL_EXPERTS + 1, dtype=np.int32)
    np.cumsum(rows_per_expert, out=expert_offsets[1:])

    # route_reverse in packed-row order
    route_reverse = np.empty(total, dtype=[("src_rank", np.int32), ("src_token", np.int32),
                                           ("topk_slot", np.int32), ("route_weight", np.float32)])
    for i, (_le, sr, st, ks, wt) in enumerate(rows):
        route_reverse[i] = (sr, st, ks, wt)

    # route_segment: one contiguous run per (expert, src_rank). Single-rank -> one per expert.
    segments = []
    for le in range(N_LOCAL_EXPERTS):
        rc = int(rows_per_expert[le])
        if rc == 0:
            continue
        dst = int(expert_offsets[le])
        segments.append((le, my_rank, 0, dst, rc))  # src_row_begin=0 (single-source synthetic)
    route_segment = np.array(segments,
                             dtype=[("expert_id", np.int32), ("src_rank", np.int32),
                                    ("src_row_begin", np.int32), ("dst_row_begin", np.int32),
                                    ("row_count", np.int32)])

    m_pad = int(np.ceil(max(total, 1) / m_pad_to) * m_pad_to) if scale_layout == SCALE_GROUP_MAJOR else 0

    params = dict(magic=ROUTE_ABI_MAGIC, version=ROUTE_ABI_VERSION, ep_size=EP_SIZE,
                  my_rank=my_rank, n_local_experts=N_LOCAL_EXPERTS, top_k=TOP_K,
                  hidden=HIDDEN, group=GROUP, n_groups=N_GROUPS, fp8_is_ocp=1,
                  scale_layout=int(scale_layout), m_pad=m_pad,
                  total_packed_rows=total, t_local=t_local)

    remote = int(np.sum((topk_ids < lo) | (topk_ids >= lo + N_LOCAL_EXPERTS)))
    return dict(params=params, topk_ids=topk_ids, topk_weights=topk_weights,
                rows_per_expert=rows_per_expert, expert_offsets=expert_offsets,
                route_segment=route_segment, route_reverse=route_reverse,
                remote_assignments=remote, max_expert_load=int(rows_per_expert.max(initial=0)))


def validate(buf):
    """Assert the correctness invariants from ROUTE_CAPTURE_SCHEMA.md 5."""
    p = buf["params"]
    rpe, off, rev = buf["rows_per_expert"], buf["expert_offsets"], buf["route_reverse"]
    assert off[0] == 0 and off[-1] == int(rpe.sum()), "expert_offsets prefix-sum broken"
    assert np.all(np.diff(off) >= 0), "expert_offsets not monotonic"
    assert off[-1] == p["total_packed_rows"], "total_packed_rows mismatch"
    assert len(rev) == p["total_packed_rows"], "route_reverse length mismatch"
    if len(rev):
        assert rev["topk_slot"].min() >= 0 and rev["topk_slot"].max() < TOP_K
        assert rev["src_rank"].min() >= 0 and rev["src_rank"].max() < EP_SIZE
    return True


def write_bin(path, bufs):
    """Write a route_capture.bin (file header + one summary record per buf, FULL arrays)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", ROUTE_ABI_MAGIC, ROUTE_ABI_VERSION, len(bufs), 1))
        # summary records (must match route_capture_record layout in route_abi.h)
        for idx, b in enumerate(bufs):
            p = b["params"]
            f.write(struct.pack("<IIiiqiiiiiiii",
                                ROUTE_ABI_MAGIC, ROUTE_ABI_VERSION, p["my_rank"], 0, 0,
                                p["t_local"], TOP_K, N_LOCAL_EXPERTS,
                                p["t_local"] * TOP_K, b["remote_assignments"], 0,
                                b["max_expert_load"], 0))
            f.write(b["rows_per_expert"].astype("<i4").tobytes())
            f.write(b["expert_offsets"].astype("<i4").tobytes())
        # FULL arena: route_reverse per record
        for idx, b in enumerate(bufs):
            rev = b["route_reverse"]
            f.write(struct.pack("<III", idx, 2, len(rev)))  # kind 2 = route_reverse
            f.write(rev.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Synthetic IRISX MoE route buffers (CPU/numpy).")
    ap.add_argument("--kind", choices=["uniform", "zipf", "hot", "many-empty"], default="uniform")
    ap.add_argument("--t-local", type=int, default=32)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hot-expert", type=int, default=0)
    ap.add_argument("--n-empty", type=int, default=200)
    ap.add_argument("--out-bin", default=None, help="write route_capture.bin")
    ap.add_argument("--out-json", default=None, help="write summary JSON")
    args = ap.parse_args()

    buf = generate(args.kind, args.t_local, args.rank, args.seed,
                   args.hot_expert, args.n_empty)
    validate(buf)
    p = buf["params"]
    print(f"kind={args.kind} rank={args.rank} t_local={args.t_local} "
          f"packed_rows={p['total_packed_rows']} max_load={buf['max_expert_load']} "
          f"remote={buf['remote_assignments']} m_pad={p['m_pad']} "
          f"nonempty_experts={int((buf['rows_per_expert']>0).sum())}/{N_LOCAL_EXPERTS}")

    if args.out_bin:
        write_bin(args.out_bin, [buf])
        print(f"wrote {args.out_bin}")
    if args.out_json:
        j = dict(magic="RAX1", version=1, records=[dict(
            rank=args.rank, layer=0, step=0, t_local=args.t_local, top_k=TOP_K,
            total_assignments=args.t_local * TOP_K,
            remote_assignments=buf["remote_assignments"], dropped_assignments=0,
            max_expert_load=buf["max_expert_load"],
            rows_per_expert=buf["rows_per_expert"].tolist(),
            expert_offsets=buf["expert_offsets"].tolist())])
        with open(args.out_json, "w") as f:
            json.dump(j, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
