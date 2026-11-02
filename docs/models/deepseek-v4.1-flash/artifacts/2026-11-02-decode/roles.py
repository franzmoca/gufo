"""Per-role kernel rates in decode steps (AR rows = 1). argv: prof_dir [request index -3..-1]"""
import sys, collections, statistics as stt
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from steps import load, requests, LOGIT
BPE = {8: 34/32, 14: 210/256, 13: 176/256, 19: 50/256, 16: 66/256, 22: 82/256, 17: 74/256, 18: 98/256, 10: 84/256, 12: 144/256}
D, F = 5120, 2304
ROLE = {  # (kernel, grid_x) -> (role, elements)
 ('gemv_kernel<8, 4, false, 1>', 57344): ('attn q_a+kv', 1792*D), ('gemv_kernel<14, 4, false, 1>', 57344): ('attn q_a+kv', 1792*D),
 ('gemv_kernel<8, 4, false, 1>', 1048576): ('attn q_b', 32768*1280), ('gemv_kernel<14, 2, false, 1>', 524288): ('attn q_b', 32768*1280),
 ('gemv_kernel<8, 8, false, 1>', 524288): ('attn output_a', 8*1024*4096), ('gemv_kernel<14, 8, false, 1>', 524288): ('attn output_a', 8*1024*4096),
 ('gemv_kernel<8, 8, false, 1>', 327680): ('attn output_b', D*8192), ('gemv_kernel<14, 8, false, 1>', 327680): ('attn output_b', D*8192),
 ('gemv_kernel<8, 4, true, 1>', 73728): ('shared gate+up', 2*F*D), ('gemv_kernel<13, 4, true, 1>', 73728): ('shared gate+up', 2*F*D),
 ('gemv_kernel<8, 2, false, 1>', 81920): ('shared down', D*F), ('gemv_kernel<13, 2, false, 1>', 81920): ('shared down', D*F),
 ('gemv_kernel<14, 8, false, 1>', 1638400): ('engram projection', 25600*6144),
 ('gemv_kernel<14, 8, false, 1>', 8273920): ('output head', 129280*D),
}
def typ(name):
    try: return int(name.split('<')[1].split(',')[0])
    except Exception: return None
def main(d, ri=-3):
    K, C = load(d)
    R = requests(K, C)[ri]
    lg = [x for x in C if R[0] <= x[0] <= R[1] and x[2] == LOGIT]
    t0, t1 = lg[1][1], lg[-1][1]
    steps = len(lg) - 2
    agg = collections.defaultdict(list)
    for k in K:
        if not (t0 <= k[0] < t1): continue
        n, gx, gy = k[2], k[3], k[4]
        key = None
        if (n, gx) in ROLE and gy == 1:
            role, el = ROLE[(n, gx)]; key = (f"{role} ({n.split('<')[1].split(',')[0]})", el * BPE[typ(n)])
        elif n.startswith('expert_gemv_kernel') and gy == 6:
            t = typ(n); el = 2*F*D*6 if gx == 18432 else D*F*6
            key = (f"routed {'gate+up' if gx == 18432 else 'down'} type {t}", el * BPE[t])
        elif n == 'matmul_bf16_vec8_kernel' and gx == 12288: key = ('router BF16', 384*D*2)
        elif n.startswith('hc_mix_small'): key = ('hc mix (FP32 weights)', 24*4*D*4)
        if key: agg[key].append(k[1] - k[0])
    print(f"{d} request {ri}: {steps} steps")
    tot = tot_ceiling = 0
    for (role, by), v in sorted(agg.items(), key=lambda x: -sum(x[1])):
        med = stt.median(v); per = sum(v) / steps / 1e6
        bw = by / med  # GB/s
        ceil = by / 240e9 * 1e3 * len(v) / steps
        tot += per; tot_ceiling += ceil
        print(f"  {role:34s} {len(v)/steps:6.2f}/tok  median {med/1e3:7.1f} us  {by/1e6:7.2f} MB  {bw:6.1f} GB/s  {per:6.3f} ms/tok  at 240: {ceil:6.3f}")
    print(f"  total {tot:.2f} ms/tok, at 240 GB/s {tot_ceiling:.2f}")
if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else -3)
