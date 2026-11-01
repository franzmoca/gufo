#!/usr/bin/env python3
"""Fit the simulator's prediction and prefetch-byte costs to live A/B ratios.

Each --case is TRACE:LIVE_RATIO, the measured decode time of the candidate
rule over the current rule (B / A) on the trace's workload.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefetch_sim import Policy, fit_read_cost, read_trace, simulate  # noqa: E402

CURRENT = Policy("current", lead=1, k=6, only_after_miss=True)
CANDIDATE = Policy("per-layer switch", lead=1, k=6, max_experts=4, switch_on=0.1, switch_off=0.07, per_layer=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    cases = []
    for item in args.case:
        path, ratio = item.rsplit(":", 1)
        snap, prepares = read_trace(Path(path))
        cases.append((path, snap, prepares, fit_read_cost(prepares), float(ratio)))
    best = None
    for pred_us in (30, 60, 100, 150, 200):
        for byte_ms_mib in (0.0, 0.01, 0.02, 0.04, 0.08):
            err, ratios = 0.0, []
            for path, snap, prepares, (a, b), live in cases:
                kw = dict(pred_cost=pred_us * 1e3, byte_cost=byte_ms_mib * 1e6 / 2**20)
                ta = simulate(snap, prepares, CURRENT, a, b, **kw)[0]
                tb = simulate(snap, prepares, CANDIDATE, a, b, **kw)[0]
                ratios.append(tb / ta)
                err += (tb / ta - live) ** 2
            print(f"pred {pred_us:3d} us, prefetch {byte_ms_mib:.2f} ms/MiB: B/A "
                  + " ".join(f"{r:.4f}" for r in ratios) + f"  (err {err:.2e})", flush=True)
            if best is None or err < best[0]:
                best = (err, pred_us, byte_ms_mib)
    print(f"best: {best[1]} us per prediction, {best[2]:.2f} ms per prefetched MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
