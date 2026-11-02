#!/usr/bin/env python3
"""Eviction rules on a DS41 expert trace (DS41_EXPERT_TRACE), on
prefetch_sim.py's cache model without prefetch: the store's score and
decay/bonus variants, LRU, per-expert reuse-interval prediction, and
Belady's offline optimum (evict the expert whose next use is farthest), with
and without its admission bypass.

  evict_sim.py TRACE                      # the default rule set
  evict_sim.py TRACE current,lru,belady   # named rules
  evict_sim.py TRACE 'grid:1/4;2/8'       # multiple of prefetch_sim.DECAY / recency bonus
  evict_sim.py TRACE 'reuse:reuse-b/4000/0.25'  # kind / first gap / gap rate
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefetch_sim as ps

class Rule(ps.Cache):
    def __init__(self, snap, kind, prepares=None, **kw):
        pol = ps.CachePolicy(kind, **{k: v for k, v in kw.items() if k in ("decay", "recent_bonus", "count_cap")})
        super().__init__(snap, None, pol)
        self.kind = kind
        self.i = -1
        self.g0 = kw.get("g0", 4000.0)
        self.grate = kw.get("grate", 0.25)
        self.gap = np.full((snap.n_layer, snap.n_expert), np.nan)
        if kind in ("belady", "belady-admit"):
            L, E = snap.n_layer, snap.n_expert
            occ = {}
            for i, p in enumerate(prepares):
                for e in np.unique(p.ids[p.ids >= 0]):
                    occ.setdefault((p.layer, int(e)), []).append(i)
            self.occ = {k: np.array(v) for k, v in occ.items()}
            self.nxt = np.full((L, E), 1 << 40, np.int64)
            for (l, e), v in self.occ.items():
                self.nxt[l, e] = v[0]

    def value(self, sl, se):
        if self.kind == "lru":
            return self.last[sl, se].astype(float)
        if self.kind.startswith("reuse"):
            g = np.where(np.isnan(self.gap[sl, se]), self.g0, self.gap[sl, se])
            age = self.tick - self.last[sl, se]
            if self.kind == "reuse-a":
                nxt = self.last[sl, se] + g
            elif self.kind == "reuse-d":
                nxt = self.last[sl, se] + g + age
            else:  # reuse-b: overdue experts are due one gap from now
                nxt = np.where(age < g, self.last[sl, se] + g, self.tick + g)
            return -nxt.astype(float)
        if self.kind in ("belady", "belady-admit"):
            return -self.nxt[sl, se].astype(float)
        return self.score[sl, se] / (1.0 + (self.tick - self.last[sl, se]) / self.policy.decay)

    def prepare(self, layer, ids):
        self.i += 1
        if self.kind.startswith("reuse"):
            for e in np.unique(ids[ids >= 0]):
                e = int(e)
                if self.last[layer, e] > 0:
                    d = self.tick + 1 - self.last[layer, e]
                    g = self.gap[layer, e]
                    self.gap[layer, e] = d if np.isnan(g) else g + self.grate * (d - g)
        out = super().prepare(layer, ids)
        if self.kind in ("belady", "belady-admit"):
            for e in np.unique(ids[ids >= 0]):
                v = self.occ[(layer, int(e))]
                j = np.searchsorted(v, self.i, side="right")
                self.nxt[layer, int(e)] = v[j] if j < len(v) else 1 << 40
        return out

    def acquire(self, layer, expert, allow_evict):
        own = self.layer_class[layer]
        if not self.free[own] and self.grow_failure[own] != self.tick and not self.grow(own):
            self.grow_failure[own] = self.tick
        for c in range(own, len(self.free)):
            if self.free[c]:
                return self.free[c].pop()
        if not allow_evict:
            return None
        live = (self.s_layer >= 0) & (self.s_class >= own)
        if self.routed is not None and self.routed[0] == layer:
            live &= ~((self.s_layer == layer) & self.routed[1][np.maximum(self.s_expert, 0)])
        if not live.any():
            return None
        idx = np.nonzero(live)[0]
        vals = self.value(self.s_layer[idx], self.s_expert[idx])
        j = int(np.argmin(vals))
        if self.kind != "belady-admit" and self.value(np.array([layer]), np.array([expert]))[0] < vals[j]:
            return None
        victim = int(idx[j])
        self.slot_of[self.s_layer[victim], self.s_expert[victim]] = -1
        return victim

def evaluate(snap, prepares, cache, a, b):
    size = snap.expert_bytes
    miss = routes = 0; dbytes = pbytes = 0; dstall = pstall = 0.0
    for p in prepares:
        missing, admit, hits = cache.prepare(p.layer, p.ids)
        mask = p.ids >= 0
        by = len(missing) * size[p.layer]
        overlap = (p.t_end - p.t_routes) - p.read_wait
        st = max(0.0, (a + b * by if by else 0.0) - overlap)
        if p.n <= 16:
            miss += int((~hits & mask).sum()); routes += int(mask.sum()); dbytes += by; dstall += st
        else:
            pbytes += by; pstall += st
        cache.admit(p.layer, missing, admit)
    return miss / routes, dbytes, pbytes, dstall, pstall

def main():
    snap, prepares = ps.read_trace(Path(sys.argv[1]))
    a, b = ps.fit_read_cost(prepares)
    D = ps.DECAY
    rules = [("current", {}), ("decay /2", dict(decay=D / 2)), ("decay /4", dict(decay=D / 4)),
             ("decay /8", dict(decay=D / 8)), ("bonus 8", dict(recent_bonus=8.0)),
             ("bonus 16", dict(recent_bonus=16.0)), ("decay /2 bonus 8", dict(decay=D / 2, recent_bonus=8.0)),
             ("decay /4 bonus 8", dict(decay=D / 4, recent_bonus=8.0)), ("lru", {}), ("belady", {}), ("belady-admit", {})]
    if len(sys.argv) > 2 and sys.argv[2].startswith("reuse:"):
        rules = [("current", {})]
        for spec in sys.argv[2][6:].split(";"):
            k, g0, gr = spec.split("/")
            rules.append((f"{k} g0 {g0} rate {gr}", dict(kind=k, g0=float(g0), grate=float(gr))))
    elif len(sys.argv) > 2 and sys.argv[2].startswith("grid:"):
        rules = []
        for spec in sys.argv[2][5:].split(";"):
            d, bo = (float(x) for x in spec.split("/"))
            rules.append((f"decay x{d:g} bonus {bo:g}", dict(decay=D * d, recent_bonus=bo)))
    elif len(sys.argv) > 2:
        rules = [r for r in rules if r[0] in sys.argv[2].split(",")]
    ref = None
    for name, kw in rules:
        kind = kw.pop("kind", None) or (name if name in ("lru", "belady", "belady-admit") else "score")
        c = Rule(snap, kind, prepares, **kw)
        m, db, pb, ds, pst = evaluate(snap, prepares, c, a, b)
        ref = ref or (m, ds, pst)
        print(f"{name:20s} decode misses {100*m:5.2f}% ({100*(m/ref[0]-1):+6.1f}%)  reads decode {db/2**30:6.1f} GiB "
              f"prefill {pb/2**30:6.1f} GiB  stall decode {ds/1e9:6.2f} s ({(ds-ref[1])/1e9:+5.2f}) "
              f"prefill {pst/1e9:6.2f} s ({(pst-ref[2])/1e9:+5.2f})", flush=True)

main()
