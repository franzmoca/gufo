import collections
import itertools
import re
import sqlite3
import sys

db, ntok = sys.argv[1], float(sys.argv[2])
c = sqlite3.connect(db)
ks = c.execute("select start, end, name from kernels order by start").fetchall()
last = max(e for s, e, n in ks if "routed_f16" in n)
ks = [k for k in ks if k[0] > last]
short = lambda n: re.sub(
    r"\(.*", "", re.sub(r"^void |\(anonymous namespace\)::", "", n)
)[:40]
g = collections.Counter()
hist = collections.Counter()
tot = 0
for (s0, e0, n0), (s1, e1, n1) in itertools.pairwise(ks):
    gap = (s1 - max(e0, s0)) / 1e3
    if gap <= 0:
        continue
    tot += gap
    b = (
        "<5us"
        if gap < 5
        else "5-20us"
        if gap < 20
        else "20-100us"
        if gap < 100
        else "100us-1ms"
        if gap < 1000
        else ">1ms"
    )
    hist[b] += gap
    if gap >= 20:
        g[f"{short(n0)} -> {short(n1)}"] += gap
print(
    f"idle {tot / ntok / 1e3:.2f} ms/token; by gap size (ms/token):",
    {k: round(v / ntok / 1e3, 2) for k, v in hist.items()},
)
for k, v in g.most_common(8):
    print(f"  {v / ntok / 1e3:6.2f} ms/token  {k}")
