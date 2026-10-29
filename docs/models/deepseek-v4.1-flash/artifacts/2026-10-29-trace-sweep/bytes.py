import collections
import glob
import os
import re
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, f):
        self.f = f

    def rd(self, fmt):
        return struct.unpack("<" + fmt, self.f.read(struct.calcsize("<" + fmt)))

    def rstr(self):
        n = self.rd("Q")[0]
        return self.f.read(n).decode("utf8", "replace")

    def rval(self, t):
        if t == 8:
            return self.rstr()
        if t == 9:
            at = self.rd("I")[0]
            n = self.rd("Q")[0]
            return [self.rval(at) for _ in range(n)]
        return self.f.read(SIZES[t])


def tensors(pat):
    out = {}
    for p in sorted(glob.glob(pat)):
        with open(p, "rb") as f:
            r = Reader(f)
            f.read(8)  # magic and version
            nt, nkv = r.rd("QQ")
            align = 32
            for _ in range(nkv):
                k = r.rstr()
                v = r.rval(r.rd("I")[0])
                if k == "general.alignment":
                    align = struct.unpack("<I", v)[0]
            info = []
            for _ in range(nt):
                name = r.rstr()
                nd = r.rd("I")[0]
                dims = r.rd("Q" * nd)
                typ = r.rd("I")[0]
                off = r.rd("Q")[0]
                info.append([name, dims, typ, off])
            data = (f.tell() + align - 1) // align * align
            end = os.path.getsize(p) - data
            info.sort(key=lambda x: x[3])
            for i, (name, dims, typ, off) in enumerate(info):
                nxt = info[i + 1][3] if i + 1 < len(info) else end
                out[name] = (dims, typ, nxt - off)
    return out


root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/models/ds41"
for label, pat in [("IQ1_M", "IQ1_M"), ("IQ2_XXS", "IQ2_XXS")]:
    t = tensors(
        f"{root}/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-{pat}-0000*-of-00006.gguf"
    )
    cat = collections.Counter()
    for name, (dims, typ, b) in t.items():
        if name in ("token_embd.weight", "per_layer_token_embd.weight"):
            cat["embd (host, 1 row)"] += 0
            continue
        if "_exps" in name:
            cat["routed experts (6/384)"] += b * 6 / dims[-1]
        elif "shexp" in name:
            cat["shared expert"] += b
        elif "attn" in name or "indexer" in name or "compressor" in name:
            cat["attention"] += b
        elif name.startswith("output"):
            cat["output head"] += b
        elif "gate_inp" in name or "exp_probs" in name or "ffn_gate." in name:
            cat["router"] += b
        elif "hc" in name:
            cat["hyperconnections"] += b
        else:
            cat["other:" + re.sub(r"blk\.\d+\.", "", name)] += b
    tot = sum(cat.values())
    print(
        f"{label}: total weight bytes per decoded token {tot / 2**20:.1f} MiB -> AR ceiling {240e9 / tot:.1f} tok/s at 240 GB/s"
    )
    for k, v in cat.most_common(14):
        print(f"   {k:40s} {v / 2**20:8.1f} MiB")
