import sys, collections; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from steps import *
d=sys.argv[1]; ri=int(sys.argv[2])
K,C=load(d)
R=requests(K,C)[ri]
lg=[x for x in C if R[0]<=x[0]<=R[1] and x[2]>0 and x[2]%LOGIT==0]
Ks=[k for k in K if R[0]<=k[0]<=R[1]]; Cs=[x for x in C if R[0]<=x[0]<=R[1]]
agg=collections.defaultdict(lambda:[0,0]); n=0; span=0
for a,b in zip(lg[:-1],lg[1:]):
    t0,t1=a[1],b[1]
    if t1-t0>1e9: continue
    ks=[k for k in Ks if t0<=k[0]<t1]
    m=[k for k in ks if k[2].startswith(('markov','top8'))]
    if not m: continue
    dl=[x for x in Cs if x[0]>=m[-1][1] and x[0]<t1]
    dend=dl[0][1] if dl else m[-1][1]
    first=[k for k in ks if not k[2].startswith('__amd')][0][0]
    n+=1; span+=dend-first
    for k in ks:
        if k[1]<=dend:
            key=k[2][:45]+(' head' if k[3]==8273920 else '')
            agg[key][0]+=1; agg[key][1]+=k[1]-k[0]
print(f"{d} req {ri}: {n} drafting cycles, drafter span (first kernel..token download) {span/n/1e6:.2f} ms")
tot=0
for kname,(c,t) in sorted(agg.items(),key=lambda x:-x[1][1])[:18]:
    tot+=t; print(f"  {c/n:6.1f}/cycle {t/n/1e3:8.1f} us  {kname}")
print(f"  listed {tot/n/1e6:.2f} ms")
