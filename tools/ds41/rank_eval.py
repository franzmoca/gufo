#!/usr/bin/env python3
"""Evaluate the built-in expert ranking (expert_rank.inc) on DS41 expert traces.

Static coverage: the share of a session's routes whose expert is among a
ranking's first N experts, against an oracle ranking from the session itself
(the bound for any fixed list), a ranking learned from the session's first
half (scored on the second half), and random per-layer orders. Dynamic: the
session replayed from caches seeded by each ranking, plus an empty cache.
"""
import argparse
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefetch_sim import Cache, Snapshot, read_trace  # noqa: E402

RANK_INC = Path(__file__).resolve().parents[2] / "src/models/deepseek_v41_flash/runtime/expert_rank.inc"
N_LAYER, N_EXPERT = 40, 384


def builtin_rank():
    text = RANK_INC.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return np.array([int(x) for x in re.findall(r"\d+", text)], np.int64)


def frequency(prepares, decode=None, part=None):
    counts = np.zeros(N_LAYER * N_EXPERT, np.int64)
    items = prepares
    if part is not None:
        half = len(prepares) // 2
        items = prepares[:half] if part == 0 else prepares[half:]
    for p in items:
        if decode is not None and (p.n <= 16) != decode:
            continue
        ids = p.ids[p.ids >= 0]
        np.add.at(counts, p.layer * N_EXPERT + ids, 1)
    return counts


def order_by(counts, rng):
    # Ties (unused experts) in random order, so no ranking gets them for free.
    return np.lexsort((rng.random(len(counts)), -counts))


def coverage(order, counts, n):
    top = np.zeros(len(counts), bool)
    top[order[:n]] = True
    return counts[top].sum() / max(counts.sum(), 1)


def seeded_snapshot(snap: Snapshot, order, budget):
    """The cache the store would seed from `order` (seed_ranked): slots in
    ranked order until the budget, every seeded score 8, nothing else."""
    slots, used = [], 0
    for index in order:
        layer, expert = divmod(int(index), N_EXPERT)
        c = snap.layer_class[layer]
        if used + snap.class_bytes[c] > budget:
            break
        used += snap.class_bytes[c]
        slots.append((c, layer, expert))
    score = np.zeros((N_LAYER, N_EXPERT))
    for _, layer, expert in slots:
        score[layer, expert] = 8.0
    return Snapshot(snap.n_layer, snap.n_expert, snap.used, 0, snap.cache_limit, used, snap.available,
                    snap.class_bytes, snap.layer_class, snap.expert_bytes, slots, score,
                    np.zeros((N_LAYER, N_EXPERT), np.int64)), len(slots)


def replay(snap, prepares, limit, first_tokens=1000):
    cache = Cache(snap, limit)
    miss = routes = early_miss = early_routes = tokens = 0
    for p in prepares:
        if p.layer == 0:
            tokens += p.n
        missing, admit, hits = cache.prepare(p.layer, p.ids)
        mask = p.ids >= 0
        m, r = int((~hits & mask).sum()), int(mask.sum())
        miss += m
        routes += r
        if tokens <= first_tokens:
            early_miss += m
            early_routes += r
        cache.admit(p.layer, missing, admit)
    return miss / routes, early_miss / max(early_routes, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--limit-gib", type=float)
    args = parser.parse_args()
    snap, prepares = read_trace(args.trace)
    limit = int(args.limit_gib * 2**30) if args.limit_gib else snap.cache_limit
    rng = np.random.default_rng(1)
    builtin = builtin_rank()
    seeded = sum(1 for s in snap.slots if s[1] != 0xFFFFFFFF)
    print(f"seeded experts in the trace: {seeded} of {N_LAYER * N_EXPERT}")

    # Seed survival and use.
    in_seed = {(s[1], s[2]) for s in snap.slots if s[1] != 0xFFFFFFFF}
    used = frequency(prepares)
    used_seeded = sum(1 for (l, e) in in_seed if used[l * N_EXPERT + e])
    print(f"seeded experts routed at least once: {used_seeded} ({100 * used_seeded / max(seeded, 1):.1f}%); "
          f"experts routed at all: {int((used > 0).sum())}")

    print("\nstatic coverage of routes by each ranking's first N experts (decode / prefill):")
    dec, pre = frequency(prepares, True), frequency(prepares, False)
    first_half = frequency(prepares, None, 0)
    dec2, pre2 = frequency(prepares, True, 1), frequency(prepares, False, 1)
    rand = np.concatenate([rng.permutation(N_EXPERT) + l * N_EXPERT for l in range(N_LAYER)])
    rand = rand.reshape(N_LAYER, N_EXPERT).T.reshape(-1)  # interleave layers
    rows = [("built-in", builtin), ("oracle (this session)", order_by(dec + pre, rng)),
            ("random per layer", rand)]
    for n in sorted({seeded, 8000, 12000, 14000}):
        print(f"  N = {n}")
        for name, order in rows:
            print(f"    {name:24s} {100 * coverage(order, dec, n):5.1f}% / {100 * coverage(order, pre, n):5.1f}%")
        learned = order_by(first_half, rng)
        print(f"    {'learned on 1st half':24s} {100 * coverage(learned, dec2, n):5.1f}% / "
              f"{100 * coverage(learned, pre2, n):5.1f}%  (2nd half; built-in there "
              f"{100 * coverage(builtin, dec2, n):5.1f}% / {100 * coverage(builtin, pre2, n):5.1f}%)")

    print("\nreplay from each seed (route misses over the session / its first 1,000 tokens):")
    budget = limit // 16 * 15  # seed_ranked: 15/16 of the cap (free memory exceeded it at load)
    seeds = [("built-in", builtin), ("oracle (this session)", order_by(dec + pre, rng)),
             ("learned on 1st half", order_by(first_half, rng)), ("random per layer", rand),
             ("empty", np.array([], np.int64))]
    for name, order in seeds:
        s, n = seeded_snapshot(snap, order, budget)
        total, early = replay(s, prepares, limit)
        print(f"  {name:24s} seeds {n:5d}: {100 * total:5.2f}% / {100 * early:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
