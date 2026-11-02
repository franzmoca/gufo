"""Verification cost by rows: step wall (logits->logits) split into drafter, kernels, read idle."""
import sys, collections, statistics as st; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from steps import *
from budget import merged
for d in sys.argv[1:]:
    K,C=load(d)
    by=collections.defaultdict(list)
    for R in requests(K,C)[-3:]:
        lg=[x for x in C if R[0]<=x[0]<=R[1] and x[2]>0 and x[2]%LOGIT==0]
        Ks=[k for k in K if R[0]<=k[0]<=R[1]]
        i=0
        for a,b in zip(lg[:-1],lg[1:]):
            t0,t1=a[1],b[1]
            if t1-t0>1e9: continue
            while i<len(Ks) and Ks[i][0]<t0: i+=1
            j=i; ks=[]
            while j<len(Ks) and Ks[j][0]<t1: ks.append(Ks[j]); j+=1
            m=[k for k in ks if k[2].startswith(('markov','top8'))]
            dend=m[-1][1] if m else t0
            ver=[k for k in ks if k[0]>=dend]
            exp=sum(k[1]-k[0] for k in ver if k[2].startswith('expert_'))
            busy=sum(e-s for s,e,_ in merged([(k[0],k[1],0) for k in ver]))
            by[b[2]//LOGIT].append(((t1-t0)/1e6,(dend-t0)/1e6,(t1-dend)/1e6,busy/1e6,exp/1e6))
    print(d.split('/')[-1])
    for r,v in sorted(by.items()):
        if len(v)<5: continue
        f=lambda i: st.mean(x[i] for x in v)
        print(f"  rows {r:2d} n={len(v):4d} cycle {f(0):6.1f} ms = pre-verify {f(1):5.1f} + verify {f(2):6.1f} (busy {f(3):5.1f}, experts {f(4):5.1f})")
