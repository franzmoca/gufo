"""Per-token decode weight bytes by tensor role from GGUF headers (all shards)."""
import struct, sys, glob, re, collections
# ggml type -> (block elems, block bytes)
T = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),10:(256,84),11:(256,110),12:(256,144),
     13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),
     22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),29:(256,56),30:(1,2),39:(32,17)}
def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
def rs(f): (n,) = rd(f, "<Q"); return f.read(n).decode()
def skip_val(f, t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.read(sz[t]); return
    if t == 8: rs(f); return
    if t == 9:
        (at,) = rd(f, "<I"); (n,) = rd(f, "<Q")
        for _ in range(n): skip_val(f, at)
        return
    raise ValueError(t)
tensors = []
for path in sorted(glob.glob(sys.argv[1])):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"; rd(f, "<I"); nt, nkv = rd(f, "<QQ")
        for _ in range(nkv): rs(f); (t,) = rd(f, "<I"); skip_val(f, t)
        for _ in range(nt):
            name = rs(f); (nd,) = rd(f, "<I"); dims = rd(f, "<%dQ" % nd); (typ,) = rd(f, "<I"); rd(f, "<Q")
            n = 1
            for d in dims: n *= d
            be, bb = T[typ]; tensors.append((name, typ, dims, n // be * bb))
groups = collections.defaultdict(lambda: [0, 0])
for name, typ, dims, b in tensors:
    if "_exps" in name: g = "routed experts (all)"
    elif name.startswith("token_embd"): g = "token embedding (host)"
    elif name == "output.weight": g = "output head"
    elif "engram" in name: g = "engram " + re.sub(r"blk\.\d+\.", "", name)
    elif "shexp" in name: g = "shared expert"
    elif "attn" in name or "indexer" in name or "compress" in name: g = "attention (" + re.sub(r"blk\.\d+\.", "", name).split(".")[0] + ")"
    elif "ffn_gate_inp" in name: g = "router"
    elif name.startswith("blk"): g = "other " + re.sub(r"blk\.\d+\.", "", name)
    else: g = "other " + name
    groups[g][0] += b; groups[g][1] += 1
for g, (b, c) in sorted(groups.items(), key=lambda x: -x[1][0]):
    print(f"{b/2**20:12.1f} MiB  {c:5d}  {g}")
exp = [t for t in tensors if "_exps" in t[0]]
per_layer = collections.defaultdict(int)
for name, typ, dims, b in exp: per_layer[int(name.split(".")[1])] += b // dims[-1]
print("routed per token (6/layer):", sum(6 * v for v in per_layer.values()) / 2**20, "MiB")
