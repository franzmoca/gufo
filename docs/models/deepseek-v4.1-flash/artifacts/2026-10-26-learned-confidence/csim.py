import re,glob,sys,math
from collections import defaultdict
V={1:41.8,2:55.1,3:67.2,4:77.8,5:87.1,6:99.1}
DRAFT=9.3; HOST=0.6
def load(tag):
    data=defaultdict(list)
    for f in sorted(glob.glob(f'/tmp/ds41-review/ctrace/*-{tag}*.err')):
        prompt=f.split('/')[-1].split('-')[0]; rows=[]
        for l in open(f):
            m=re.search(r'conf row=(\d+) top=([\d.]+) learned=([-\d.]+)',l)
            if m:
                if int(m[1])==0: rows=[]
                rows.append((float(m[2]),float(m[3]))); continue
            m=re.search(r'DSpark pos=\d+ drafted=(\d+) accepted=(\d+)',l)
            if m and int(m[1])==5 and len(rows)==5: data[prompt].append((rows,int(m[2]))); rows=[]
    return data
def auc(pairs):
    pos=[s for s,y in pairs if y]; neg=[s for s,y in pairs if not y]
    if not pos or not neg: return float('nan')
    allv=sorted([(s,1) for s in pos]+[(s,0) for s in neg]); rank=0; r=0;i=0
    # Mann-Whitney with ties
    vals=[v for v,_ in allv]; ranks={}
    i=0
    while i<len(allv):
        j=i
        while j<len(allv) and allv[j][0]==allv[i][0]: j+=1
        for k in range(i,j): ranks.setdefault(k,(i+j+1)/2)
        i=j
    sp=sum(ranks[k] for k,(v,y) in enumerate(allv) if y)
    return (sp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def run(cycles,choose):
    tok=ms=0
    for rows,a in cycles:
        k=choose(rows)
        tok+=min(a,k)+1; ms+=V[k+1]+DRAFT+HOST*k
    return tok/ms*1000
def first(f):
    def c(rows):
        k=0
        for r in rows:
            if not f(r): break
            k+=1
        return k
    return c
def ev(sig):
    # include position while expected tokens/ms improves (V4-style, cumulative survival)
    def c(rows):
        best_k=0; best=1/(V[1]+DRAFT); surv=1; exp_=1
        for k,r in enumerate(rows,1):
            surv*=min(max(sig(r),0),1); exp_+=surv
            rate=exp_/(V[k+1]+DRAFT+HOST*k)
            if rate>best: best=rate; best_k=k
        return best_k
    return c
for tag in ('s','g'):
    data=load(tag); allc=[c for v in data.values() for c in v]
    pairs_t=[];pairs_l=[];pairs_p=[]
    bins=defaultdict(lambda:[0,0])
    for rows,a in allc:
        for i,(t,l) in enumerate(rows):
            if i>a: break
            y=i<a; pairs_t.append((t,y)); pairs_l.append((l,y)); pairs_p.append((t*l,y))
            b=min(int(l*10),9); bins[b][0]+=1; bins[b][1]+=y
    print(f'== {"sampled" if tag=="s" else "greedy"} cycles={len(allc)} rows={len(pairs_l)}  AUC qtop={auc(pairs_t):.3f} learned={auc(pairs_l):.3f} product={auc(pairs_p):.3f}')
    print('  learned calibration (bin: n, observed):',' '.join(f'{b/10:.1f}:{n},{acc/n:.2f}' for b,(n,acc) in sorted(bins.items())))
    pols={'w=0 (plain)':lambda r:0, 'w=5':lambda r:5}
    for w in (1,2,3): pols[f'w={w}']=(lambda w: lambda r:w)(w)
    pols['qtop>=0.6']=first(lambda r:r[0]>=0.6)
    for tau in (0.3,0.4,0.5,0.6,0.7): pols[f'learned>={tau}']=first((lambda tau: lambda r:r[1]>=tau)(tau))
    pols['EV learned']=ev(lambda r:r[1])
    pols['EV qtop*learned^.5']=ev(lambda r:r[0]*r[1]**.5)
    pols['qtop>=.6 & learned>=.4']=first(lambda r:r[0]>=0.6 and r[1]>=0.4)
    print('  '+'policy'.ljust(24)+''.join(p.rjust(9) for p in data)+'      all')
    for n,c in pols.items():
        if tag=='g' and 'qtop' in n: continue
        print('  '+n.ljust(24)+''.join(f'{run(v,c):9.1f}' for v in data.values())+f'{run(allc,c):9.1f}')
print('== extra')
for tag in ('s','g'):
    data=load(tag); allc=[c for v in data.values() for c in v]
    pols={}
    for tau in (0.45,0.5,0.55): pols[f'learned>={tau}']=first((lambda tau: lambda r:r[1]>=tau)(tau))
    for tau in (0.2,0.3,0.4,0.5):
        def surv(tau=tau):
            def c(rows):
                s=1;k=0
                for r in rows:
                    s*=r[1]
                    if s<tau: break
                    k+=1
                return k
            return c
        pols[f'survival>={tau}']=surv()
    if tag=='s':
        pols['learned>=.5 & qtop>=.6']=first(lambda r:r[1]>=0.5 and r[0]>=0.6)
        pols['learned>=.5 | qtop>=.95']=first(lambda r:r[1]>=0.5 or r[0]>=0.95)
    print(tag,' '.join(f'{n}={run(allc,c):.1f}' for n,c in pols.items()))
