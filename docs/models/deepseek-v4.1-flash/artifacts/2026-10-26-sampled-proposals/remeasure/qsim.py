import re,glob,sys
from collections import defaultdict
V={1:41.8,2:55.1,3:67.2,4:77.8,5:87.1,6:99.1}
draft=lambda k: 8.3+0.87*k
data=defaultdict(list)  # prompt -> cycles [(tops, accepted)]
for f in sorted(glob.glob('/tmp/ds41-review/qtrace/*.err')):
    prompt=f.split('/')[-1].split('-')[0]; tops=[]
    for l in open(f):
        m=re.search(r'q row=(\d+) top=([\d.]+) size=(\d+)',l)
        if m:
            if int(m[1])==0: tops=[]
            tops.append(float(m[2]) if int(m[3])>1 else 1.0); continue
        m=re.search(r'DSpark pos=\d+ drafted=(\d+) accepted=(\d+)',l)
        if m and int(m[1])==5 and len(tops)==5: data[prompt].append((tops,int(m[2]))); tops=[]
def run(cycles,choose):
    tok=ms=0
    for tops,a in cycles:
        k=choose(tops)
        tok+=min(a,k)+1; ms+=V[k+1]+(draft(5) if getattr(choose,"drafts",True) else 0)
    return tok/ms*1000
def fw(w):
    f=lambda t: w
    f.drafts=w>0
    return f
fixed={f'w={w}':fw(w) for w in range(0,6)}
def thr(tau):
    def c(t):
        k=0
        for x in t:
            if x<tau: break
            k+=1
        return k
    return c
def cum(tau):
    def c(t):
        k=0;pr=1
        for x in t:
            pr*=x
            if pr<tau: break
            k+=1
        return k
    return c
pols=dict(fixed)
for tau in (0.2,0.3,0.4,0.5,0.6,0.7): pols[f'first<{tau}']=thr(tau)
for tau in (0.1,0.2,0.3,0.4,0.5): pols[f'cum<{tau}']=cum(tau)
print('policy'.ljust(12),*[p.rjust(8) for p in data],'   all')
allc=[c for v in data.values() for c in v]
for name,c in pols.items():
    print(name.ljust(12),*[f'{run(v,c):8.1f}' for v in data.values()],f'{run(allc,c):8.1f}')
print('cycles',{p:len(v) for p,v in data.items()})
# acceptance vs confidence bins, conditional on prefix accepted
import math
bins=defaultdict(lambda:[0,0])
for tops,a in allc:
    for i,x in enumerate(tops):
        if i>a: break
        b=min(int(x*10),9); bins[b][0]+=1; bins[b][1]+= i<a
print('conf bin: n accept-rate', {f'{b/10:.1f}':(n,round(acc/n,2)) for b,(n,acc) in sorted(bins.items())})
