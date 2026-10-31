import re,sys,os,statistics as st
exec(open('replay2.py').read().split("pols=[")[0])  # loads logs, V, D, P, policies
PRI=[42.4,69.6,83.6,95.5,106.1,117.2]
def cost(w): return P if w==0 else D+V[w]
def best(cm,conf):
    bw,b=0,1/cm[0];reach=1;e=1
    for w in range(1,6):
        reach*=conf[w-1];e+=reach
        if e/cm[w]>b:b,bw=e/cm[w],w
    return bw
def sim(rows,update):
    cm=PRI[:];tok=0;t=0
    for conf,L,*_ in rows:
        w=best(cm,conf); c=cost(w)
        tok+=1+min(w,L) if w else 1; t+=c+ (D if w==0 else 0)
        update(cm,w,c)
    return tok/t*1000
def per_width(cm,w,c):
    if w: cm[w]+=0.1*(c-cm[w])
def shift(cm,w,c):
    if w:
        d=c-cm[w]; cm[w]+=0.1*d
        for k in range(1,6):
            if k!=w: cm[k]+=0.05*d
def affine(cm,w,c,st_={}):
    # EMA least squares of t = a + b*w over drafted cycles, prior pseudo-points
    s=st_.setdefault(id(cm),{'n':5.0,'sw':15.0,'sww':55.0,'st':sum(PRI[1:]),'swt':sum(k*PRI[k] for k in range(1,6))})
    if not w: return
    r=0.1
    for key,val in (('n',1),('sw',w),('sww',w*w),('st',c),('swt',w*c)): s[key]=(1-r)*s[key]+r*val*5
    b=(s['n']*s['swt']-s['sw']*s['st'])/(s['n']*s['sww']-s['sw']**2); a=(s['st']-b*s['sw'])/s['n']
    for k in range(1,6): cm[k]=a+b*k
def oracle_cost(cm,w,c):
    for k in range(1,6): cm[k]=cost(k)
res={}
for name,u in [("per_width",per_width),("shift",shift),("affine",affine),("true_costs",oracle_cost)]:
    vals=[sim(rows,u) for rows,_ in logs.values()]
    res[name]=st.geometric_mean(vals)
base=st.geometric_mean([run(rows,thresh(.5)) for rows,_ in logs.values()])
print("th.5",round(base,2)); 
for k,v in res.items(): print(k,round(v,2),f"{100*(v/base-1):+.2f}%")
