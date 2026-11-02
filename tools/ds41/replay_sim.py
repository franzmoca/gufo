#!/usr/bin/env python3
"""Decoder-replay read stalls on a DS41 expert trace (DS41_EXPERT_TRACE).

A prompt's last batch runs layers 20-39 on its last rows only (the decoder
replay); each of those layers waits on reads. The trace records, for those
rows, the router's top-12 predictions for every replay layer from the last
encoder layer's FFN input. This tool reports:

  A. how much of each replay layer's missing experts those predictions hold;
  B. the replay stall if the predicted experts were read from the moment
     the last encoder layer ends (one read queue at the fitted rate);
  C. replay, prompt and decode reads and stalls under cache policies that
     keep the replay's experts cached.

Uses prefetch_sim.py's trace reader and cache model.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefetch_sim as ps  # noqa: E402

FIRST_DECODER = 20
REPLAY_MAX = 140


def replays(prepares):
    """(index of the last encoder layer's prepare, [indices of layers 20-39]) per prompt replay."""
    out = []
    for i, p in enumerate(prepares):
        if p.layer != FIRST_DECODER - 1 or p.n <= REPLAY_MAX:
            continue
        group = []
        for j in range(i + 1, min(len(prepares), i + 1 + (ps_layers(prepares) - FIRST_DECODER))):
            q = prepares[j]
            if q.layer != FIRST_DECODER + len(group) or q.n > REPLAY_MAX:
                break
            group.append(j)
        if group:
            out.append((i, group))
    return out


def ps_layers(prepares):
    return max(p.layer for p in prepares) + 1


@dataclass
class Protect:
    """Victim filter: decoder-layer experts used by a batch of at most
    REPLAY_MAX rows within `window` ticks are not evicted by prompt batches."""
    window: int = 0           # 0: off; < 0: protect every decoder-layer expert
    only_prompt: bool = True  # only prompt-sized admissions are restricted


class ProtectCache(ps.Cache):
    def __init__(self, snap, limit=None, policy=ps.CachePolicy(), protect=Protect()):
        super().__init__(snap, limit, policy)
        self.protect = protect
        self.small_last = np.full((snap.n_layer, snap.n_expert), -10**9, np.int64)
        self.batch_n = 0

    def prepare(self, layer, ids):
        self.batch_n = ids.shape[0]
        missing, admit, hits = super().prepare(layer, ids)
        if self.batch_n <= REPLAY_MAX:
            used = np.unique(ids[ids >= 0])
            self.small_last[layer, used] = self.tick
        return missing, admit, hits

    def acquire(self, layer, expert, allow_evict):
        pr = self.protect
        if pr.window == 0 or (pr.only_prompt and self.batch_n <= REPLAY_MAX):
            return super().acquire(layer, expert, allow_evict)
        # Temporarily hide protected slots from the victim search.
        live_layer = self.s_layer.copy()
        dec = live_layer >= FIRST_DECODER
        if pr.window > 0:
            recent = np.zeros_like(dec)
            idx = np.nonzero(dec)[0]
            recent[idx] = self.tick - self.small_last[live_layer[idx], self.s_expert[idx]] <= pr.window
            dec &= recent
        hidden = np.nonzero(dec)[0]
        saved = self.s_layer[hidden].copy()
        self.s_layer[hidden] = -2           # neither live nor free
        try:
            slot = super().acquire(layer, expert, allow_evict)
        finally:
            self.s_layer[hidden] = saved
        return slot


def kind(prepares, i, replay_set):
    if i in replay_set:
        return "replay"
    return "decode" if prepares[i].n <= 16 else "prompt"


def evaluate(snap, prepares, a, b, policy, protect, limit, replay_set):
    cache = ProtectCache(snap, limit, policy, protect)
    size = snap.expert_bytes
    out = {k: [0, 0.0, 0.0] for k in ("replay", "prompt", "decode")}  # bytes, stall, recorded wait
    for i, p in enumerate(prepares):
        missing, admit, _ = cache.prepare(p.layer, p.ids)
        bytes_ = len(missing) * size[p.layer]
        overlap = (p.t_end - p.t_routes) - p.read_wait
        stall = max(0.0, (a + b * bytes_ if bytes_ else 0.0) - overlap)
        k = kind(prepares, i, replay_set)
        out[k][0] += bytes_
        out[k][1] += stall
        out[k][2] += p.read_wait
        cache.admit(p.layer, missing, admit)
    return out


def recall(snap, prepares, groups, ks=(6, 8, 12)):
    """Per replay layer: missing experts under the production cache and the
    share each top-k prediction from the last encoder layer holds."""
    cache = ps.Cache(snap)
    want = {j: (i, j) for i, g in groups for j in g}
    res = {k: [0, 0, 0] for k in ks}   # caught, missing, extra (predicted, uncached, unused)
    missing_at = {}
    for idx, p in enumerate(prepares):
        missing, admit, _ = cache.prepare(p.layer, p.ids)
        if idx in want:
            lead = p.layer - (FIRST_DECODER - 1)
            pred = p.pred.get(lead)
            if pred is not None:
                miss = set(int(e) for e in missing)
                used = set(int(e) for e in np.unique(p.ids[p.ids >= 0]))
                for k in ks:
                    predicted = set(int(e) for e in np.unique(pred[:, :k]))
                    res[k][0] += len(miss & predicted)
                    res[k][1] += len(miss)
                    res[k][2] += len({e for e in predicted - used if not cache.resident(p.layer, e)})
            missing_at[idx] = missing
        cache.admit(p.layer, missing, admit)
    return res, missing_at


def readahead(snap, prepares, groups, missing_at, a, b, k):
    """Replay stall when the top-k predicted uncached experts of every replay
    layer are read from the end of the last encoder layer's prepare, in layer
    order on one queue, and unpredicted misses are demand reads."""
    size = snap.expert_bytes
    base = sim = 0.0
    for i, g in groups:
        p19 = prepares[i]
        busy = float(p19.t_end)                    # queue free when the last encoder layer's reads end
        ready = {}
        for j in g:
            q = prepares[j]
            pred = q.pred.get(q.layer - (FIRST_DECODER - 1))
            if pred is None:
                continue
            predicted = set(int(e) for e in np.unique(pred[:, :k]))
            for e in sorted(predicted & set(int(x) for x in missing_at[j])):
                busy += b * size[q.layer]
                ready[j, e] = busy
            # wasted reads also occupy the queue
            wasted = len(predicted - set(int(x) for x in np.unique(q.ids[q.ids >= 0])))
            busy += b * size[q.layer] * wasted * 0  # counted separately in recall(); kept free here
        t = float(prepares[g[0]].t_routes)
        prev_end = None
        for j in g:
            q = prepares[j]
            gap = 0.0 if prev_end is None else max(0.0, q.t_routes - prev_end)
            t += gap
            compute = (q.t_end - q.t_routes) - q.read_wait
            done = t
            demand = [e for e in missing_at[j] if (j, e) not in ready]
            for e in missing_at[j]:
                if (j, e) in ready:
                    done = max(done, ready[j, e])
            if demand:
                start = max(t, busy)
                busy = start + a + b * size[q.layer] * len(demand)
                done = max(done, busy)
            stall = max(0.0, done - (t + compute))
            sim += stall
            base += q.read_wait
            t += compute + stall
            prev_end = q.t_end
    return base, sim


POLICIES = [
    ("current", ps.CachePolicy(), Protect()),
    ("protect all decoder layers in prompts", ps.CachePolicy(), Protect(window=-1)),
    ("protect decoder experts used <= 500 ticks ago", ps.CachePolicy(), Protect(window=500)),
    ("protect decoder experts used <= 2000 ticks ago", ps.CachePolicy(), Protect(window=2000)),
    ("protect decoder experts used <= 10000 ticks ago", ps.CachePolicy(), Protect(window=10000)),
    ("prompt admits n/64", ps.CachePolicy("n/64", admit_div=64), Protect()),
    ("prompt admits n/128", ps.CachePolicy("n/128", admit_div=128), Protect()),
    ("n/64 + protect <= 2000", ps.CachePolicy("n/64", admit_div=64), Protect(window=2000)),
]

ADMISSION = [
    ("current (n/32, >= 4)", ps.CachePolicy(), Protect()),
    ("prompt admits n/64", ps.CachePolicy("n/64", admit_div=64), Protect()),
    ("prompt admits n/128", ps.CachePolicy("n/128", admit_div=128), Protect()),
    ("prompt admits n/256", ps.CachePolicy("n/256", admit_div=256), Protect()),
    ("prompt admits n/512", ps.CachePolicy("n/512", admit_div=512), Protect()),
    ("prompt admits final rows only", ps.CachePolicy("final", admit_div=10**9, admit_min=10**9), Protect()),
    ("prompt admits all", ps.CachePolicy("all", admit_all=True), Protect()),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--read-cost", default="0.38,0.10",
                        help="read cost 'ms,ms_per_MiB' (default: PREFILL.md's fit for the two drives)")
    parser.add_argument("--limit-gib", type=float)
    parser.add_argument("--admission", action="store_true", help="only sweep the prompt admission rule (C)")
    args = parser.parse_args()
    a_ms, b_ms = (float(x) for x in args.read_cost.split(","))
    a, b = a_ms * 1e6, b_ms * 1e6 / 2**20
    limit = int(args.limit_gib * 2**30) if args.limit_gib else None
    snap, prepares = ps.read_trace(args.trace)
    ps.summary(snap, prepares)
    agree, total, first_bad = ps.validate(snap, prepares, limit)
    print(f"cache replay: {100 * agree / max(total, 1):.3f}% of route hits reproduced")
    groups = replays(prepares)
    replay_set = {j for _, g in groups for j in g}
    wait = sum(prepares[j].read_wait for j in replay_set)
    print(f"{len(groups)} decoder replays, {len(replay_set)} layer prepares, recorded read wait {wait / 1e9:.2f} s")

    if args.admission:
        print("\nC. prompt admission (estimated stalls)")
        ref = None
        for name, pol, prot in ADMISSION:
            r = evaluate(snap, prepares, a, b, pol, prot, limit, replay_set)
            ref = ref or r
            total = sum(v[1] for v in r.values()) - sum(v[1] for v in ref.values())
            print(f"  {name:32s} " + " | ".join(
                f"{k} {r[k][0] / 2**30:6.1f} GiB {r[k][1] / 1e9:5.2f} s ({(r[k][1] - ref[k][1]) / 1e9:+5.2f})"
                for k in ("replay", "prompt", "decode")) + f" | total {total / 1e9:+5.2f} s")
        return 0
    print("\nA. prediction from the last encoder layer (all replay layers)")
    res, missing_at = recall(snap, prepares, groups)
    for k, (caught, miss, extra) in res.items():
        print(f"  top-{k:2d}: {100 * caught / max(miss, 1):5.1f}% of {miss} missing experts; "
              f"{extra} extra uncached experts read ({100 * extra / max(miss, 1):.0f}% of the misses)")
    by_lead = {}
    cache_res = None
    print("\nB. read-ahead from the end of the last encoder layer (one queue)")
    for k in (6, 8, 12):
        base, sim = readahead(snap, prepares, groups, missing_at, a, b, k)
        print(f"  top-{k:2d}: replay stall {base / 1e9:.2f} s recorded -> {sim / 1e9:.2f} s")

    print("\nC. cache policies (estimated stalls; recorded waits for reference)")
    ref = None
    for name, pol, prot in POLICIES:
        r = evaluate(snap, prepares, a, b, pol, prot, limit, replay_set)
        ref = ref or r
        cells = []
        for k in ("replay", "prompt", "decode"):
            by, st, _ = r[k]
            cells.append(f"{k} {by / 2**30:6.1f} GiB {st / 1e9:5.2f} s ({(st - ref[k][1]) / 1e9:+5.2f})")
        print(f"  {name:48s} " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
