#!/usr/bin/env python3
"""CPU oracle for the DeepSeek V4.1 Flash DSpark drafter.

Recomputes one draft from the inputs ``ds41_load_probe --dspark-dump DIR`` writes
(the target features of the last prompt rows and the anchor), following the
official inference/model.py DSparkBlock: main_x = main_norm(main_proj(features)),
per-stage window rings of kv_norm(wkv(main_x)) (roped, FP8), block rows
[anchor, noise...] through the three stages, the target head and the Markov path.
Reports how closely the GPU head logits and drafts match.

    python3 -I tools/ds41/dspark_reference.py TARGET-00001-of-N.gguf DRAFTER.gguf DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gguf_model import Model, Tensor, TensorFile  # noqa: E402
from reference import (fp8_fake_quant, hc_split_sinkhorn, linear, precompute_freqs_cis, rms_norm,  # noqa: E402
                       rotate_tail)

NOISE = 128799
BLOCK = 5
STAGES = 3
HC = 4
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
LIMIT = 10.0
ROUTE_SCALE = 1.5
TOPK = 3
WINDOW = 128


class Drafter:
    def __init__(self, path: Path):
        source = TensorFile(path)
        self.tensors = {r["name"]: Tensor(source, r) for r in source.info["tensors"]}
        self.lucebox = source.kv.get("general.architecture") == "deepseek41-dflash-draft"
        self.cache: dict[str, torch.Tensor] = {}

    def name(self, key: str) -> str:
        if not self.lucebox:
            return key
        return {"fc.weight": "dflash.fc.weight", "enc.output_norm.weight": "dflash.hidden_norm.weight",
                "markov_w1.weight": "dflash.dspark.markov.w1", "markov_w2.weight": "dflash.dspark.markov.w2"}.get(key, key)

    def t(self, key: str) -> torch.Tensor:
        key = self.name(key)
        if key not in self.cache:
            self.cache[key] = torch.from_numpy(self.tensors[key].array()).float()
        return self.cache[key]

    def expert(self, stage: int, kind: str, e: int) -> torch.Tensor:
        return torch.from_numpy(self.tensors[f"blk.{stage}.ffn_{kind}_exps.weight"].array(e, 1)[0]).float()


def mixes(h, fn, scale, base):
    x = h.flatten(1)
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return hc_split_sinkhorn(F.linear(x, fn) * rsqrt, scale, base, HC, SINKHORN, HC_EPS)


def swiglu_expert(x, gate_w, up_w, down_w, weight=None):
    gate = linear(x, gate_w).clamp(max=LIMIT)
    up = linear(x, up_w).clamp(-LIMIT, LIMIT)
    h = F.silu(gate) * up
    if weight is not None:
        h = weight * h
    return linear(h, down_w)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path)
    parser.add_argument("drafter", type=Path)
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    meta = [int(v) for v in (args.dump / "draft.txt").read_text().split()]
    pos0, rows, anchor, gpu_drafts = meta[0], meta[1], meta[2], meta[3:3 + BLOCK]
    features = torch.from_numpy(np.fromfile(args.dump / "features.f32", dtype=np.float32).reshape(rows, -1))
    gpu_logits = torch.from_numpy(np.fromfile(args.dump / "logits.f32", dtype=np.float32).reshape(BLOCK, -1))
    target = Model(args.target)
    d = Drafter(args.drafter)
    pos = pos0 + rows  # the anchor's position
    freqs = precompute_freqs_cis(64, pos + BLOCK, 0, 10000.0, 1.0, 32, 1, torch.device("cpu"))
    head_dim, heads, rd = 512, 64, 64
    first = max(pos0, pos - WINDOW)  # the dump holds the ring rows the GPU drafted with

    main_x = rms_norm(linear(features, d.t("fc.weight")), d.t("enc.output_norm.weight"), EPS)
    main_x = main_x[first - pos0:]
    rings = []
    for s in range(STAGES):
        kv = rms_norm(linear(main_x, d.t(f"blk.{s}.attn_kv.weight")), d.t(f"blk.{s}.attn_kv_a_norm.weight"), EPS)
        rings.append(fp8_fake_quant(rotate_tail(kv, freqs[first:pos], rd)))

    embed = target["token_embd.weight"]
    tokens = [anchor] + [NOISE] * (BLOCK - 1)
    x = torch.from_numpy(np.stack([embed.array(t, 1)[0] for t in tokens])).float()
    h = x.unsqueeze(1).repeat(1, HC, 1)
    pre = torch.zeros(BLOCK, HC)
    pre[:, 0] = 1.0
    block_freqs = freqs[pos:pos + BLOCK]
    for s in range(STAGES):
        p = f"blk.{s}."
        residual = h
        attn_pre, post, comb = mixes(h, d.t(p + "hc_attn_fn.weight"), d.t(p + "hc_attn_scale.weight"),
                                     d.t(p + "hc_attn_base.weight"))
        x = rms_norm((pre.unsqueeze(-1) * h).sum(1), d.t(p + "attn_norm.weight"), EPS)
        qr = rms_norm(linear(x, d.t(p + "attn_q_a.weight")), d.t(p + "attn_q_a_norm.weight"), EPS)
        q = rotate_tail(linear(qr, d.t(p + "attn_q_b.weight")).unflatten(-1, (heads, head_dim)), block_freqs, rd)
        kv = rms_norm(linear(x, d.t(p + "attn_kv.weight")), d.t(p + "attn_kv_a_norm.weight"), EPS)
        keys = torch.cat([rings[s], fp8_fake_quant(rotate_tail(kv, block_freqs, rd))])
        scores = torch.einsum("shd,kd->shk", q, keys) * head_dim**-0.5
        sink = d.t(p + "attn_sinks.weight").view(1, -1, 1)
        top = torch.maximum(scores.amax(-1, keepdim=True), sink)
        weights = torch.exp(scores - top)
        o = torch.einsum("shk,kd->shd", weights / (weights.sum(-1, keepdim=True) + torch.exp(sink - top)), keys)
        o = fp8_fake_quant(rotate_tail(o, block_freqs, rd, inverse=True).reshape(BLOCK, 8, -1))
        o = torch.einsum("sgd,grd->sgr", o, d.t(p + "attn_output_a.weight").view(8, 1024, -1))
        x = linear(o.flatten(1), d.t(p + "attn_output_b.weight"))
        h = post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(1)

        residual = h
        ffn_pre, post, comb = mixes(h, d.t(p + "hc_ffn_fn.weight"), d.t(p + "hc_ffn_scale.weight"),
                                    d.t(p + "hc_ffn_base.weight"))
        x = rms_norm((attn_pre.unsqueeze(-1) * h).sum(1), d.t(p + "ffn_norm.weight"), EPS)
        scores = F.softplus(linear(x, d.t(p + "ffn_gate_inp.weight"))).sqrt()
        idx = (scores + d.t(p + "exp_probs_b.bias")).topk(TOPK, dim=-1)[1]
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        y = torch.zeros_like(x)
        for e in sorted(set(idx.flatten().tolist())):
            r, slot = torch.where(idx == e)
            y[r] += swiglu_expert(x[r], d.expert(s, "gate", e), d.expert(s, "up", e), d.expert(s, "down", e),
                                  w[r, slot, None])
        y += swiglu_expert(x, d.t(p + "ffn_gate_shexp.weight"), d.t(p + "ffn_up_shexp.weight"),
                           d.t(p + "ffn_down_shexp.weight"))
        h = post.unsqueeze(-1) * y.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(1)
        pre = ffn_pre

    x = rms_norm((pre.unsqueeze(-1) * h).sum(1), d.t("output_norm.weight"), EPS)
    output = torch.from_numpy(target["output.weight"].array()).float()
    logits = F.linear(x, output)
    w1, w2 = d.t("markov_w1.weight"), d.t("markov_w2.weight")
    drafts, prev = [], anchor
    for i in range(BLOCK):
        prev = int((logits[i] + F.linear(w1[prev], w2)).argmax())
        drafts.append(prev)

    print(f"anchor {anchor} at position {pos}; ring rows {first}..{pos - 1}")
    for i in range(BLOCK):
        ref, gpu = logits[i], gpu_logits[i]
        cos = F.cosine_similarity(ref, gpu, dim=0).item()
        print(f"row {i}: max |diff| {(ref - gpu).abs().max().item():.4f} (logit range {ref.abs().max().item():.2f}), "
              f"cosine {cos:.6f}, argmax ref {int(ref.argmax())} gpu {int(gpu.argmax())}")
    print(f"drafts ref {drafts}\ndrafts gpu {gpu_drafts}")
    return 0 if drafts == gpu_drafts else 1


if __name__ == "__main__":
    sys.exit(main())
