"""Per-token decode budget per request: kernel classes, idle attribution. argv: prof_dir"""
import sys, collections, statistics as stt
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from steps import load, requests, LOGIT
OUT_HEAD, ENGRAM = 8273920, 1638400
DRAFT_MARK = ('markov_', 'top8_bias', 'gather_row')
def cls(k):
    n, gx = k[2], k[3]
    if n.startswith('__amd') or n.startswith('gufo::hip'): return 'copy kernels'
    if 'gemv_kernel' in n and 'expert' not in n:
        if gx == OUT_HEAD: return 'output head'
        if gx == ENGRAM: return 'engram'
        return 'dense GEMV'
    if n.startswith('expert_'): return 'routed experts'
    if n.startswith('engram'): return 'engram'
    if n.startswith('matmul_bf16') or n.startswith('route_kernel'): return 'router/BF16'
    if n.startswith('hc_'): return 'hyper-connections'
    if any(n.startswith(m) for m in DRAFT_MARK) or n.startswith('gather_row') or 'dspark' in n or 'confidence' in n: return 'drafter-only'
    if n.startswith('argmax_rows'): return 'argmax'
    return 'attention+misc'
def merged(iv):
    iv = sorted(iv); out = []
    for s, e, tag in iv:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e, tag])
    return out
def analyse(d, label=''):
    K, C = load(d)
    R = requests(K, C)
    reqs = R[-3:]
    for name, (r0, r1) in zip(['chat', 'code', 'long'], reqs):
        lg = [x for x in C if r0 <= x[0] <= r1 and x[2] > 0 and x[2] % LOGIT == 0]
        # decode steps: after the first logits download (prefill end)
        Ks = [k for k in K if r0 <= k[0] <= r1]
        Cs = [x for x in C if r0 <= x[0] <= r1]
        tot = collections.Counter(); idle = collections.Counter(); nsteps = 0; wall = 0; rows = 0
        walls = []
        ki = 0
        for a, b in zip(lg[:-1], lg[1:]):
            t0, t1 = a[1], b[1]
            ks = [k for k in Ks if t0 <= k[0] < t1]
            cs = [x for x in Cs if t0 <= x[0] < t1]
            if not ks: continue
            # skip prefill-like steps
            if any(k[2].startswith('routed_f16') for k in ks) or t1 - t0 > 1e9: continue
            nsteps += 1; wall += t1 - t0; walls.append((t1 - t0) / 1e6); rows += b[2] // LOGIT
            # drafter region: up to the last drafter-marker kernel's following download
            marks = [k for k in ks if any(k[2].startswith(m) for m in DRAFT_MARK)]
            dend = None
            if marks:
                lm = marks[-1][1]
                dl = [x for x in cs if x[0] >= lm]
                dend = dl[0][1] if dl else lm
            for k in ks:
                c = cls(k)
                if dend and k[1] <= dend: c = 'drafter (+inject)'
                tot[c] += k[1] - k[0]
            for x in cs:
                tot['memcpy: expert uploads' if x[2] >= 1 << 20 else 'memcpy: small'] += x[1] - x[0]
            iv = [(k[0], k[1], k) for k in ks if k[5] == 0 or True] + [(x[0], x[1], ('copy', x)) for x in cs]
            m = merged(iv)
            busy = sum(e - s for s, e, _ in m)
            tot['_busy'] += busy
            prev = t0
            for s, e, tag in m:
                g = s - prev
                if g > 0:
                    if prev == t0: c = 'idle: step start (host)'
                    elif dend and s <= dend + 1: c = 'idle: drafter gaps'
                    elif isinstance(tag, tuple) and tag[0] == 'copy':
                        sz = tag[1][2]
                        if g <= 20e3: c = 'idle: launch gaps <20us'
                        elif sz % 24576 == 0 and sz <= 24576 * 16: c = 'idle: engram rows'
                        elif sz >= 1 << 20: c = 'idle: expert reads (upload next)'
                        else: c = f'idle: before copy {sz}B'
                    elif g <= 20e3: c = 'idle: launch gaps <20us'
                    elif tag[2].startswith('expert_'): c = 'idle: expert reads'
                    elif tag[2].startswith('gemv') and tag[3] == ENGRAM: c = 'idle: engram'
                    else: c = 'idle: other >20us (' + tag[2][:28] + ')'
                    idle[c] += g
                prev = max(prev, e)
            if t1 > prev: idle['idle: step end'] += t1 - prev
        if not nsteps: continue
        toks = rows  # rows evaluated (AR: tokens)
        print(f"\n## {label} {name}: {nsteps} steps, {rows} rows, wall {wall/1e9:.2f} s, {wall/nsteps/1e6:.2f} ms/step "
              f"(median {stt.median(walls):.2f}), busy {tot['_busy']/nsteps/1e6:.2f} ms/step")
        for c, v in sorted(tot.items(), key=lambda x: -x[1]):
            if c != '_busy': print(f"   {v/nsteps/1e6:7.3f} ms  {c}")
        for c, v in sorted(idle.items(), key=lambda x: -x[1])[:10]:
            print(f"   {v/nsteps/1e6:7.3f} ms  {c}")
if __name__ == '__main__':
    analyse(sys.argv[1], sys.argv[1].split('/')[-1])
