"""Builds the int4 W4A4 prefill sidecar for Gemma 4 31B from the BF16
checkpoint (docs/models/gemma-4-31b/WEIGHT-FORMATS.md, "Integration
design").

Recipe: per layer and projection input, x' = H(D x) with D random signs and
H the orthonormal Walsh-Hadamard transform over blocks of 128; the weights
get the same transform on their rows (W x = W' x') and GPTQ (1% damping,
columns in order) to symmetric int4 with one binary16 scale per 128
inputs, on Hessians of calibration-v6 (bartowski). The quantizer is the
simulator's (quant.py), so the file holds exactly the weights the
`w4a4gptq-c6` variant evaluates.

Tensors per layer (GGUF projection names; gate and up stay separate, as in
the model GGUF, and fuse at load):

    blk.N.<group>.w4a4.vec     F32 [K]           D, per input of
                               attn_qkv, attn_output, ffn_gate_up, ffn_down
    blk.N.<proj>.w4a4.codes    I32 [K/128][M][16] int4 two's complement,
                               eight per little-endian word, even k in the
                               low nibble, K-block-major
    blk.N.<proj>.w4a4.scales   F16 [K/128][M]

Activations are quantized at run time per token and per 128 rotated
inputs (gufo.w4a4.act_group).

    python -I tools/gemma4/quant_sim/sidecar.py \
        --model ~/models/gemma-4/gemma-31b-it-bf16 \
        --tokens ~/models/gemma-4/quant-sim/chat_tokens.pt \
        --hessian-cache ~/models/gemma-4/quant-sim/hessians \
        --out gemma-4-31B-it-w4a4-prefill.gguf
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/franz/projects/strix-llama.cpp/gguf-py")
import fastops  # noqa: E402
import gguf  # noqa: E402
import quant  # noqa: E402
import sim  # noqa: E402

VERSION = 1
ROT = 128
GROUP = 128
ACT_GROUP = 128
BITS = 4
DAMP = 0.01
CALIB = "v6"
GROUP_NAME = {"qkv": "attn_qkv", "o": "attn_output", "gate_up": "ffn_gate_up",
              "down": "ffn_down"}


def hessian_dir(cache, rows):
    """The simulator's cache directory for these calibration rows (exact
    FP32 or kernel-accumulated), or a new kernel one."""
    tag = hashlib.sha1(rows.numpy().tobytes()).hexdigest()[:12]
    for suffix in ("-f32", "-wmma"):
        d = cache / f"{CALIB}-{tag}{suffix}"
        if d.is_dir():
            return d
    return cache / f"{CALIB}-{tag}-wmma"


def quantize_class(li, cls, layer, hess):
    """(vec, {proj: (codes int8 [M, K], scales [M, K/128])}) for one input."""
    names = [n for n in quant.LINEARS[cls] if n in layer.w]
    k = layer.w[names[0]].shape[1]
    dev = layer.w[names[0]].device
    block = quant.rotation_block(k, ROT)
    assert block == ROT, (cls, k)
    d = quant.signs(li, cls, k, dev)
    hs = hess * d[:, None] * d[None, :]
    hr = quant.fwht(quant.fwht(hs, block).t().contiguous(),
                    block).t().contiguous()
    del hs
    w = quant.fwht(torch.cat([layer.w[n].float() for n in names]) * d, block)
    q, sc = quant.gptq(w, hr, BITS, group=GROUP, damp=DAMP, scales=True)
    del hr, w
    # Scales are binary16 values already, except the 1e-12 floor of
    # all-zero groups, which stores as 0.
    sc16 = sc.half()
    s = sc16.float().repeat_interleave(GROUP, 1)
    codes = torch.round(q / s.clamp_min(1e-30))
    codes[s == 0] = 0
    assert codes.max() <= 7 and codes.min() >= -8
    # Each weight is exactly its code times the stored scale.
    assert torch.equal(codes * s, q)
    out = {}
    row = 0
    for n in names:
        m = layer.w[n].shape[0]
        out[n] = (codes[row:row + m].to(torch.int8), sc16[row:row + m])
        row += m
    return d, out


def pack(codes, scales):
    """codes int8 [M, K] in [-8, 7] -> uint8 [K/128, M, 64]; scales
    [M, K/128] -> f16 [K/128, M]."""
    m, k = codes.shape
    c = codes.view(m, k // GROUP, GROUP).permute(1, 0, 2).to(torch.int16)
    lo = c[..., 0::2] & 0xF
    hi = c[..., 1::2] & 0xF
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()
    return packed.cpu().numpy(), scales.t().contiguous().cpu().numpy()


def collect_hessians(args, cfg, ckpt, rows, cache, dev):
    """Calibration forward through the BF16 model, kernel-accumulated input
    Hessians per layer saved to `cache` (the simulator's format)."""
    cache.mkdir(parents=True, exist_ok=True)
    embed = ckpt.get("embed_tokens.weight").to(dev)
    scale = torch.tensor(cfg.hidden ** 0.5, dtype=torch.bfloat16).float().item()
    states = [(embed[r.to(dev)].float() * scale).cpu() for r in rows]
    del embed
    rope = sim.rope_tables(cfg, rows.shape[1], dev)
    for li in range(cfg.layers):
        path = cache / f"layer{li}.pt"
        layer = sim.Layer(ckpt, cfg, li, dev)
        sums = {}

        def collect(c, inp):
            if c not in sums:
                sums[c] = torch.zeros(inp.shape[1], inp.shape[1], device=dev)
            fastops.hessian_accumulate(sums[c], inp)

        def lin(name, inp):
            return sim.half_matmul(inp, layer.w16[name])

        states = [sim.block(x.to(dev), layer, lin, cfg, rope,
                            hooks=collect).cpu() for x in states]
        if not path.exists():
            n = sum(x.shape[0] for x in states)
            torch.save({c: (fastops.symmetrize(h) / n).cpu()
                        for c, h in sums.items()}, path)
        del layer, sums
        torch.cuda.empty_cache()
        print(f"hessians: layer {li + 1}/{cfg.layers}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True,
                   help="chat_tokens.pt holding calib_v6 (calib_file.py)")
    p.add_argument("--hessian-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layers", type=int, default=0,
                   help="debug: convert only the first N layers")
    args = p.parse_args()
    dev = torch.device("cuda")
    cfg = sim.Config(args.model)
    ckpt = sim.Checkpoint(args.model)
    rows = torch.load(args.tokens)[f"calib_{CALIB}"]
    cache = hessian_dir(args.hessian_cache, rows)
    layers = args.layers or cfg.layers
    t0 = time.time()
    if not all((cache / f"layer{li}.pt").exists() for li in range(layers)):
        collect_hessians(args, cfg, ckpt, rows, cache, dev)
    print(f"hessians: {cache}", flush=True)

    config = (args.model / "config.json").read_bytes()
    w = gguf.GGUFWriter(args.out, "gemma4-w4a4-prefill")
    w.add_name("Gemma 4 31B it W4A4 prefill sidecar")
    w.add_uint32("gufo.w4a4.version", VERSION)
    w.add_string("gufo.w4a4.source", "google/gemma-4-31B-it (BF16)")
    w.add_string("gufo.w4a4.source.config_sha256",
                 hashlib.sha256(config).hexdigest())
    w.add_uint32("gufo.w4a4.block_count", layers)
    w.add_uint32("gufo.w4a4.embedding_length", cfg.hidden)
    w.add_uint32("gufo.w4a4.rotation", ROT)
    w.add_string("gufo.w4a4.rotation.kind", "signs then orthonormal "
                 "Walsh-Hadamard per block: x' = H(vec * x)")
    w.add_uint32("gufo.w4a4.weight_group", GROUP)
    w.add_uint32("gufo.w4a4.act_group", ACT_GROUP)
    w.add_string("gufo.w4a4.method", f"GPTQ, damping {DAMP}, symmetric int4, "
                 "binary16 scales, MSE clip search")
    w.add_string("gufo.w4a4.calibration",
                 f"bartowski calibration-v6, {rows.shape[0]} x "
                 f"{rows.shape[1]} tokens")
    for li in range(layers):
        lt = time.time()
        layer = sim.Layer(ckpt, cfg, li, dev)
        hessians = torch.load(cache / f"layer{li}.pt")
        for cls in quant.CLASSES:
            d, out = quantize_class(li, cls, layer, hessians[cls].to(dev))
            w.add_tensor(f"blk.{li}.{GROUP_NAME[cls]}.w4a4.vec",
                         d.float().cpu().numpy())
            for n, (codes, scales) in out.items():
                packed, sc = pack(codes, scales)
                base = f"blk.{li}.{quant.GGUF_NAME[n]}.w4a4"
                w.add_tensor(f"{base}.codes", packed.view(np.int32))
                w.add_tensor(f"{base}.scales", sc)
        del layer, hessians
        torch.cuda.empty_cache()
        print(f"layer {li + 1}/{layers} {time.time() - lt:.1f}s "
              f"(total {time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB) "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
