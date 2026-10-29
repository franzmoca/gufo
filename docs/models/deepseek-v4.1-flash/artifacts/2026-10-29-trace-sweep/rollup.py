import collections
import sqlite3
import sys

sys.path.insert(
    0,
    __import__("os").path.join(
        __import__("os").path.dirname(__file__), "../../../../../tools/prof"
    ),
)
import prof

stages = prof.STAGE_MAPS["ds41"]


def roll(db, unit, prefill):
    c = sqlite3.connect(db)
    last = (
        c.execute(
            "select max(end) from kernels where name like '%routed_f16%'"
        ).fetchone()[0]
        or 0
    )
    if prefill:  # the last pass split at idle gaps over 100 ms (both repetitions here)
        q = c.execute(
            "select start, end, duration, name from kernels order by start"
        ).fetchall()
        passes = [[]]
        for s, e, d, n in q:
            if passes[-1] and s - passes[-1][-1][1] > 100e6:
                passes.append([])
            passes[-1].append((s, e, d, n))
        ks = passes[-1]
    else:
        ks = c.execute(
            "select start, end, duration, name from kernels where start > ? order by start",
            (last,),
        ).fetchall()
    busy = sum(k[2] for k in ks) / 1e6
    span = (ks[-1][1] - ks[0][0]) / 1e6
    st = collections.Counter()
    for s, e, d, n in ks:
        st[prof.stage_of(n, stages)] += d / 1e6
    return (
        busy / unit,
        span / unit,
        len(ks) / unit,
        {k: v / unit for k, v in st.most_common()},
    )


for m in ["iq1m", "iq2xxs"]:
    for mode, unit in [
        ("pp", 1),
        ("ar-greedy", 255),
        ("ar-sampled", 255),
        ("ds-greedy", 255),
        ("ds-sampled", 255),
    ]:
        b, s, l, st = roll(
            f"{sys.argv[1]}/{m}-{mode}.prof/prof_results.db", unit, mode == "pp"
        )
        print(
            f"## {m} {mode}: busy {b:.2f} ms, span {s:.2f} ms, {l:.0f} launches per {'pass' if mode == 'pp' else 'token'}"
        )
        print("   " + ", ".join(f"{k} {v:.2f}" for k, v in st.items() if v >= 0.02 * b))
