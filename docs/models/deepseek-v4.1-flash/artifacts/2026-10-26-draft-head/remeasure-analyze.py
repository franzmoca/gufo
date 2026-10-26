#!/usr/bin/env python3
"""analyze.py FILE.jsonl BASE CAND: paired tok/s ratios plus pooled acceptance."""
import json, math, statistics, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
base, cand = sys.argv[2], sys.argv[3]
pairs = defaultdict(dict)
for r in rows: pairs[(r['rep'], r['seed'], r['prompt'])][r['label']] = r
ratios, byprompt = [], defaultdict(list)
pool = {base: defaultdict(float), cand: defaultdict(float)}
for key, d in sorted(pairs.items()):
    if base not in d or cand not in d: continue
    b, c = d[base], d[cand]
    ratio = c['tps'] / b['tps']; ratios.append(ratio); byprompt[key[2]].append(ratio)
    for lab, r in ((base, b), (cand, c)):
        for k in ('drafted', 'accepted', 'verification_steps'): pool[lab][k] += r['spec'].get(k, 0)
        pool[lab]['tokens'] += r['tokens']; pool[lab]['time'] += r['tokens'] / r['tps']
    print(f"{key[2]:8} seed={key[1]:3} rep={key[0]} {b['tps']:7.2f} -> {c['tps']:7.2f} ({100*(ratio-1):+5.1f}%)"
          f"  acc {b['spec'].get('acceptance',0):.2f}->{c['spec'].get('acceptance',0):.2f} same={b['hash']==c['hash']}")
gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))
for p, xs in byprompt.items(): print(f"{p:8} geomean {100*(gm(xs)-1):+5.2f}% n={len(xs)}")
print(f"ALL geomean {100*(gm(ratios)-1):+5.2f}%  wins {sum(r>1 for r in ratios)}/{len(ratios)}"
      f"  sd {100*statistics.stdev(ratios):.1f}%")
for lab, s in pool.items():
    print(f"{lab}: pooled tok/s {s['tokens']/s['time']:.2f} acceptance {s['accepted']/max(s['drafted'],1):.3f}"
          f" drafted {s['drafted']:.0f} accepted/step {s['accepted']/max(s['verification_steps'],1):.2f}")
