import sys, statistics as st; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from steps import *
for d in sys.argv[1:]:
    K,C=load(d)
    for name,R in zip(['chat','code','long'],requests(K,C)[-3:]):
        lg=[x for x in C if R[0]<=x[0]<=R[1] and x[2]>0 and x[2]%LOGIT==0]
        Ks=[k for k in K if R[0]<=k[0]<=R[1]]
        gaps=[];rows=[]
        i=0
        for a,b in zip(lg[:-1],lg[1:]):
            t0,t1=a[1],b[1]
            if t1-t0>1e9: continue
            while i<len(Ks) and Ks[i][0]<t0: i+=1
            # host gap: from logits copy end to first kernel that is not a copy kernel
            j=i
            while j<len(Ks) and Ks[j][2].startswith('__amd'): j+=1
            if j<len(Ks) and Ks[j][0]<t1:
                gaps.append((Ks[j][0]-t0)/1e6); rows.append(a[2]//LOGIT)
        if gaps:
            by={}
            for g,r in zip(gaps,rows): by.setdefault(r,[]).append(g)
            print(d.split('/')[-1],name,f"steps {len(gaps)} host gap mean {st.mean(gaps):.2f} ms median {st.median(gaps):.2f}",
                  ' '.join(f"rows{r}:{st.mean(v):.2f}x{len(v)}" for r,v in sorted(by.items())))
