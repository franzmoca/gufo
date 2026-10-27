import re,sys,statistics as st
def load(path):
    seq={}; cyc=[]; plains=[]
    for line in open(path):
        m=re.search(r'DSpark pos=(\d+) drafted=(\d+) accepted=(\d+) draft=([\d.]+)ms verify=([\d.]+)ms all=\d+ conf=([\d.,]+) tok=([-\d,]+) next=(-?\d+)',line)
        if m:
            p,d,a=int(m[1]),int(m[2]),int(m[3]); conf=[float(x) for x in m[6].split(',')]; tok=[int(x) for x in m[7].split(',')]
            seq[p]=tok[0]
            for i in range(a): seq[p+1+i]=tok[1+i]
            seq[p+1+a]=int(m[8]); cyc.append((p,d,a,float(m[4]),float(m[5]),conf,tok[1:6])); continue
        m=re.search(r'ds41: plain pos=(\d+) tok=(-?\d+) next=(-?\d+) ms=([\d.]+)',line)
        if m: seq[int(m[1])]=int(m[2]); seq[int(m[1])+1]=int(m[3]); plains.append(float(m[4]))
    rows=[]
    for p,d,a,dm,vm,conf,tok in cyc:
        L=0
        while L<5 and seq.get(p+1+L)==tok[L]: L+=1
        rows.append((conf,L,d,a,dm,vm))
    return rows,plains
logs={f:load(f) for f in sys.argv[1:]}
allrows=[r for rows,_ in logs.values() for r in rows]
ver={}
for r in allrows: ver.setdefault(r[2],[]).append(r[5])
V={k:st.median(v) for k,v in ver.items()}
D=st.median(r[4] for r in allrows)
P=float(__import__("os").environ.get("PLAIN","45.0"))
print("V",{k:(len(ver[k]),round(V[k],1)) for k in sorted(V)},"D",D)
def thresh(th):
    def f(conf,R):
        w=0
        while w<5 and conf[w]>=th: w+=1
        return w
    return f
def cum(th):
    def f(conf,R):
        w=0;r=1
        while w<5:
            r*=conf[w]
            if r<th: break
            w+=1
        return w
    return f
def ratio(conf,R):
    best,bw=1/(D+P),0; reach=1; e=1
    for w in range(1,6):
        reach*=conf[w-1]; e+=reach
        if e/(D+V[w])>best: best,bw=e/(D+V[w]),w
    return bw
def dink(conf,R):
    best,bw=1-R*P,0; reach=1; e=1
    for w in range(1,6):
        reach*=conf[w-1]; e+=reach
        if e-R*V[w]>best: best,bw=e-R*V[w],w
    return bw
def run(rows,policy):
    tok=0;t=0;R=0.03
    for conf,L,d,a,dm,vm in rows:
        w=policy(conf,R)
        if w==0: tok+=1;t+=D+P
        else: tok+=1+min(w,L);t+=D+V[w]
        R=tok/t
    return tok/t*1000
pols=[("th.5",thresh(.5)),("th.4",thresh(.4)),("cum.3",cum(.3)),("cum.35",cum(.35)),("cum.4",cum(.4)),("ratio",ratio),("rate",dink)]+[(f"fix{w}",(lambda c,R,w=w:w)) for w in (2,3,5)]
print("%-34s"%"log"+"".join("%8s"%n for n,_ in pols))
tot={n:[] for n,_ in pols}
for f,(rows,_) in logs.items():
    vals=[run(rows,p) for _,p in pols]
    for (n,_),v in zip(pols,vals): tot[n].append(v)
    print("%-34s"%f.split('/')[-1]+"".join("%8.2f"%v for v in vals))
print("%-34s"%"geomean"+"".join("%8.2f"%(st.geometric_mean(tot[n])) for n,_ in pols))
# calibration over all
bins=[0,.3,.4,.5,.6,.7,.8,.9,1.01]
for k in range(5):
    out=[]
    for lo,hi in zip(bins,bins[1:]):
        xs=[L>k for conf,L,*_ in allrows if L>=k and lo<=conf[k]<hi]
        if xs: out.append(f"{lo:.1f}:{sum(xs)/len(xs):.2f}({len(xs)})")
    print("pos",k+1," ".join(out))
def ratio0(conf,R):
    best,bw=1/P,0; reach=1; e=1
    for w in range(1,6):
        reach*=conf[w-1]; e+=reach
        if e/(D+V[w])>best: best,bw=e/(D+V[w]),w
    return bw
def ratio_sunk(conf,R):
    best,bw=1/P,0; reach=1; e=1
    for w in range(1,6):
        reach*=conf[w-1]; e+=reach
        if e/(V[w])>best: best,bw=e/(V[w]),w
    return bw
for n,p in [("ratio0",ratio0),("ratio_sunk",ratio_sunk)]:
    print(n, round(st.geometric_mean([run(rows,p) for rows,_ in logs.values()]),2))
