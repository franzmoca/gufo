#!/usr/bin/env python3
"""Independent DeepSeek V4.1 Flash oracle over the shipped GGUF weights.

A direct port of the official ``inference/model.py`` and ``inference/engram.py``
(deepseek-ai/DeepSeek-V4.1-Flash-1017) to plain torch, with weights dequantized
from the GGUF by ``quants.py``. It runs one sequence, exactly as the official
forward does: prefill at position 0, then single-token decode steps. Routed
experts are dequantized on demand, so a short prompt only touches the experts it
routes to. Activations stay fp32; the official FP8 window KV, FP4 compressed KV
and FP4 indexer quantization are reproduced because they change attention
inputs materially. Vision, DSpark and tensor parallelism are out of scope.

Subcommands:
  verify-engram   Rebuild the Engram layout (primes, offsets, multipliers and
                  the compressed token map) from the official algorithm and
                  compare it with the GGUF tensors.
  run             Prefill a prompt, greedily decode, and write logits, routing
                  and optional per-layer hidden states.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gguf_model import Model  # noqa: E402

DEAD = -1


@dataclass(frozen=True)
class Config:
    n_layers: int
    dim: int
    n_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window: int
    compress_ratios: tuple[int, ...]
    kv_sources: tuple[int, ...]
    index_sources: tuple[int, ...]
    index_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source: int
    candidate_blocks: int
    candidate_block_size: int
    n_experts: int
    n_active: int
    route_scale: float
    swiglu_limit: float
    norm_eps: float
    hc_mult: int
    hc_iters: int
    hc_eps: float
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    original_seq_len: int
    beta_fast: float
    beta_slow: float
    engram_layers: tuple[int, ...]
    engram_ngram: int
    engram_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab: int
    engram_vocab: int
    vocab: int

    @classmethod
    def from_model(cls, m: Model) -> "Config":
        g = m.meta
        if g("expert_scoring_func") != "sqrtsoftplus" or g("hash_layer_count", 0) != 0:
            raise ValueError("unsupported routing")
        if g("expert_shared_count") != 1 or not g("expert_weights_norm"):
            raise ValueError("unsupported expert layout")
        limits = set(g("swiglu_clamp_exp")) | set(g("swiglu_clamp_shexp"))
        if len(limits) != 1:
            raise ValueError("per-layer SwiGLU limits are not supported")
        return cls(
            n_layers=g("block_count"), dim=g("embedding_length"), n_heads=g("attention.head_count"),
            head_dim=g("attention.key_length"), rope_dim=g("rope.dimension_count"),
            q_lora_rank=g("attention.q_lora_rank"), o_groups=g("attention.output_group_count"),
            o_lora_rank=g("attention.output_lora_rank"), window=g("attention.sliding_window"),
            compress_ratios=tuple(g("attention.compress_ratios")),
            kv_sources=tuple(g("kv_source_layer_ids")), index_sources=tuple(g("index_source_layer_ids")),
            index_heads=g("attention.indexer.head_count"), index_head_dim=g("attention.indexer.key_length"),
            index_topk=g("attention.indexer.top_k"), candidate_source=g("attention.candidate_source_layer_id"),
            candidate_blocks=g("attention.candidate_topk_blocks"),
            candidate_block_size=g("attention.candidate_block_size"),
            n_experts=g("expert_count"), n_active=g("expert_used_count"),
            route_scale=g("expert_weights_scale"), swiglu_limit=limits.pop(),
            norm_eps=g("attention.layer_norm_rms_epsilon"), hc_mult=g("hyper_connection.count"),
            hc_iters=g("hyper_connection.sinkhorn_iterations"), hc_eps=g("hyper_connection.epsilon"),
            rope_theta=g("rope.freq_base"), compress_rope_theta=g("attention.compress_rope_freq_base"),
            rope_factor=g("rope.scaling.factor"), original_seq_len=g("rope.scaling.original_context_length"),
            beta_fast=g("rope.scaling.yarn_beta_fast"), beta_slow=g("rope.scaling.yarn_beta_slow"),
            engram_layers=tuple(g("engram.layer_ids", ())), engram_ngram=g("engram_max_ngram_size"),
            engram_heads=g("engram_n_heads"), engram_head_dim=g("engram_head_dim"),
            engram_pad_id=g("engram_pad_token_id"), engram_compressed_vocab=g("engram_compressed_vocab_size"),
            engram_vocab=g("engram_vocab_size"), vocab=m["output.weight"].shape[0],
        )


# --------------------------------------------------------------------------
# Quantization the official kernels apply to activations and caches
# --------------------------------------------------------------------------
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive normal x, as the kernels' exponent-bit trick computes it."""
    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(mantissa == 0.5, exponent - 1, exponent))


def fp8_fake_quant(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """act_quant(..., scale_fmt='ue8m0', inplace=True): E4M3 values, power-of-two scale per block."""
    shape = x.shape
    v = x.float().reshape(*shape[:-1], -1, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2_ceil(amax / 448.0)
    q = (v / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shape)


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 (|v| <= 6), ties to the even mantissa like a hardware cast."""
    a = v.abs().double()
    grid = FP4_GRID.to(v.device)
    distance = (a.unsqueeze(-1) - grid).abs() + (torch.arange(8, device=v.device) % 2) * 1e-12
    return (grid[distance.argmin(-1)] * torch.sign(v.double())).float()


def fp4_fake_quant(x: torch.Tensor, block: int, e4m3_scale: bool) -> torch.Tensor:
    """fp4_act_quant(..., inplace=True). E8M0 scales (indexer) or E4M3 scales (compressed KV)."""
    shape = x.shape
    v = x.float().reshape(*shape[:-1], -1, block)
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (amax.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        scale = _pow2_ceil(amax.clamp_min(6 * 2.0**-126) / 6.0)
    q = _round_fp4((v / scale).clamp(-6.0, 6.0))
    return (q * scale).reshape(shape)


# --------------------------------------------------------------------------
# Building blocks (official semantics)
# --------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def rotate(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """apply_rotary_emb on adjacent pairs; x is [s, d] or [s, h, d], freqs is [s, d/2]."""
    c = torch.view_as_complex(x.float().contiguous().unflatten(-1, (-1, 2)))
    f = freqs.conj() if inverse else freqs
    if c.ndim == 3:
        f = f.unsqueeze(1)
    return torch.view_as_real(c * f).flatten(-2)


def rotate_tail(x: torch.Tensor, freqs: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    return torch.cat([x[..., :-rd], rotate(x[..., -rd:], freqs, inverse)], dim=-1)


def hc_split_sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attention(q, kv, sink, idxs, scale, chunk=256):
    """q [s, h, d], kv [n, d] (keys and values), idxs [s, k] with -1 for empty slots."""
    out = torch.empty_like(q)
    for start in range(0, q.size(0), chunk):
        qi, ii = q[start:start + chunk], idxs[start:start + chunk].long()
        valid = ii >= 0
        gathered = kv[ii.clamp_min(0)]  # [c, k, d]
        scores = torch.einsum("chd,ckd->chk", qi, gathered) * scale
        scores = scores.masked_fill(~valid.unsqueeze(1), -torch.inf)
        top = scores.amax(-1, keepdim=True).clamp_min(-1e30)
        weights = torch.exp(scores - top)
        denom = weights.sum(-1, keepdim=True) + torch.exp(sink.view(1, -1, 1) - top)
        out[start:start + chunk] = torch.einsum("chk,ckd->chd", weights / denom, gathered)
    return out


def window_idxs(window: int, seqlen: int, start_pos: int, device) -> torch.Tensor:
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window + 1
        idxs = torch.cat([torch.arange(oldest, window), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs).unsqueeze(0)
    return idxs.int().to(device)


def candidate_blocks(logits, compress_lens, topk_blocks, block):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(-1)
    n = scores.size(-1)
    last = (compress_lens - 1) // block
    scores = scores.masked_fill(torch.arange(n, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, n), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block, dim=-1)[..., :width]


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------
class Weights:
    def __init__(self, model: Model, device: torch.device, dtype: torch.dtype, threads: int):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.cache: dict[str, torch.Tensor] = {}
        self.pool = ThreadPoolExecutor(threads)

    def get(self, name: str) -> torch.Tensor:
        value = self.cache.get(name)
        if value is None:
            array = self.model[name].array()
            storage = torch.float32 if array.ndim == 1 or array.size < 1 << 16 else self.dtype
            value = torch.from_numpy(array).to(self.device, storage)
            self.cache[name] = value
        return value

    def f(self, name: str) -> torch.Tensor:
        return self.get(name).float()

    def has(self, name: str) -> bool:
        return name in self.model

    def rows(self, name: str, ids: list[int]) -> torch.Tensor:
        tensor = self.model[name]
        return torch.from_numpy(np.stack([tensor.array(i, 1)[0] for i in ids])).to(self.device)

    def experts(self, layer: int, ids: list[int]):
        """Dequantize (gate, up, down) for the given experts in parallel; yields per expert."""
        names = [f"blk.{layer}.ffn_{p}_exps.weight" for p in ("gate", "up", "down")]
        tensors = [self.model[n] for n in names]

        def load(expert):
            return expert, [t.array(expert, 1)[0] for t in tensors]

        batch = 2 * self.pool._max_workers  # bounds host memory: ~140 MB per expert
        for start in range(0, len(ids), batch):
            for expert, arrays in self.pool.map(load, ids[start:start + batch]):
                yield expert, [torch.from_numpy(a).to(self.device) for a in arrays]


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x.float(), w.float())


# --------------------------------------------------------------------------
# Engram
# --------------------------------------------------------------------------
class EngramHash:
    """NgramHashState with the layout tensors shipped in the GGUF."""

    def __init__(self, cfg: Config, model: Model):
        self.cfg = cfg
        self.token_map = torch.from_numpy(model["engram_token_map.weight"].array().astype(np.int64))
        self.primes = torch.from_numpy(model["engram_primes.weight"].array())  # [layers, cols]
        self.offsets = torch.from_numpy(model["engram_offsets.weight"].array())  # [layers, cols]
        self.multipliers = torch.from_numpy(model["engram_multipliers.weight"].array())  # [layers, ngram]
        self.pad = int(self.token_map[cfg.engram_pad_id])
        self.history: list[int] = []

    def __call__(self, tokens: list[int]) -> torch.Tensor:
        """Hash ids [s, n_engram_layers, cols] for the new tokens."""
        start = len(self.history)
        self.history.extend(int(self.token_map[t]) for t in tokens)
        cache = torch.tensor(self.history, dtype=torch.int64)
        pos = torch.arange(start, len(self.history))
        blocked = torch.zeros_like(pos, dtype=torch.bool)
        grams = []
        for shift in range(self.cfg.engram_ngram):
            source = cache[(pos - shift).clamp_min(0)]
            blocked = blocked | (pos < shift) | (source == DEAD)
            grams.append(torch.where(blocked, self.pad, source))
        grams = torch.stack(grams, -1)  # [s, ngram]
        products = grams.unsqueeze(1) * self.multipliers  # [s, layers, ngram]
        rolling, hashes = products[..., 0], []
        heads = self.cfg.engram_heads
        for i in range(1, self.cfg.engram_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            primes = self.primes[:, (i - 1) * heads:i * heads]
            hashes.append(rolling.unsqueeze(-1) % primes)
        return torch.cat(hashes, -1) + self.offsets


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None


class Layer:
    def __init__(self, cfg: Config, w: Weights, layer_id: int, max_seq: int):
        self.cfg, self.w, self.id = cfg, w, layer_id
        self.p = f"blk.{layer_id}."
        self.ratio = cfg.compress_ratios[layer_id]
        self.is_kv_source = layer_id in cfg.kv_sources
        self.is_index_source = layer_id in cfg.index_sources
        self.has_engram = layer_id in cfg.engram_layers
        dev = w.device
        if self.ratio:
            self.freqs = precompute_freqs_cis(cfg.rope_dim, max_seq, cfg.original_seq_len, cfg.compress_rope_theta,
                                              cfg.rope_factor, cfg.beta_fast, cfg.beta_slow, dev)
        else:
            self.freqs = precompute_freqs_cis(cfg.rope_dim, max_seq, 0, cfg.rope_theta, cfg.rope_factor,
                                              cfg.beta_fast, cfg.beta_slow, dev)
        self.window_kv = torch.zeros(cfg.window, cfg.head_dim, device=dev)
        if self.is_kv_source:
            self.compress_kv = torch.zeros(max_seq // self.ratio, cfg.head_dim, device=dev)
            self.index_k = torch.zeros(max_seq // self.ratio, cfg.index_head_dim, device=dev)
            if self.ratio > 1:
                self.kv_state = torch.zeros(self.ratio, cfg.head_dim, device=dev)
                self.score_state = torch.full((self.ratio, cfg.head_dim), -torch.inf, device=dev)

    def t(self, name: str) -> torch.Tensor:
        return self.w.f(self.p + name + ".weight")

    # -- compressor -----------------------------------------------------
    def compressor(self, x: torch.Tensor, start_pos: int):
        ratio = self.ratio
        kv = linear(x, self.t("attn_compressor_kv"))
        if ratio == 1:
            return rms_norm(kv, self.t("attn_compressor_norm"), self.cfg.norm_eps)
        score = linear(x, self.t("attn_compressor_gate"))
        seqlen = x.size(0)
        if start_pos == 0:
            should = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                self.kv_state[:remainder] = kv[cutoff:]
                self.score_state[:remainder] = score[cutoff:]
            kv = kv[:cutoff].unflatten(0, (-1, ratio))
            score = score[:cutoff].unflatten(0, (-1, ratio))
            kv = (kv * score.softmax(dim=1)).sum(dim=1)
        else:
            should = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[slot] = kv[0]
            self.score_state[slot] = score[0]
            if should:
                kv = (self.kv_state * self.score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        if not should:
            return None
        return rms_norm(kv, self.t("attn_compressor_norm"), self.cfg.norm_eps)

    def group_freqs(self, seqlen: int, start_pos: int):
        ratio = self.ratio
        if start_pos == 0:
            return self.freqs[:seqlen - seqlen % ratio:ratio]
        return self.freqs[start_pos + 1 - ratio].unsqueeze(0)

    # -- indexer --------------------------------------------------------
    def indexer(self, x, qr, latent, start_pos, offset, shared: Shared):
        cfg = self.cfg
        seqlen = x.size(0)
        ratio, rd, end_pos = self.ratio, cfg.rope_dim, start_pos + seqlen
        if self.is_kv_source and latent is not None:
            k = rms_norm(linear(latent, self.t("indexer.attn_k")), self.t("indexer.k_norm"), cfg.norm_eps)
            k = rotate_tail(k, self.group_freqs(seqlen, start_pos), rd)
            k = fp4_fake_quant(k, 32, False)
            self.index_k[start_pos // ratio:start_pos // ratio + k.size(0)] = k
        if self.is_kv_source:
            shared.index_k = self.index_k
        q = linear(qr, self.t("indexer.attn_q_b")).unflatten(-1, (cfg.index_heads, cfg.index_head_dim))
        q = rotate_tail(q, self.freqs[start_pos:end_pos], rd)
        q = fp4_fake_quant(q, 32, False)
        index_k = shared.index_k[:end_pos // ratio]
        weights = linear(x, self.t("indexer.proj")) * (cfg.index_head_dim**-0.5 * cfg.index_heads**-0.5)
        score = torch.einsum("shd,td->sht", q, index_k)
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=1)
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio
        if self.id == cfg.candidate_source:
            shared.candidates = candidate_blocks(score, compress_lens, cfg.candidate_blocks, cfg.candidate_block_size)
        elif 0 <= cfg.candidate_source < self.id:
            score = score.masked_fill(~shared.candidates, -torch.inf)
        topk = min(cfg.index_topk, end_pos // ratio)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    # -- attention ------------------------------------------------------
    def attention(self, x: torch.Tensor, start_pos: int, shared: Shared) -> torch.Tensor:
        cfg = self.cfg
        seqlen = x.size(0)
        rd, win = cfg.rope_dim, cfg.window
        freqs = self.freqs[start_pos:start_pos + seqlen]
        qr = rms_norm(linear(x, self.t("attn_q_a")), self.t("attn_q_a_norm"), cfg.norm_eps)
        q = linear(qr, self.t("attn_q_b")).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        q = rotate_tail(q, freqs, rd)

        kv = rms_norm(linear(x, self.t("attn_kv")), self.t("attn_kv_a_norm"), cfg.norm_eps)
        kv = fp8_fake_quant(rotate_tail(kv, freqs, rd))
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv[:seqlen] = kv
            else:
                cutoff = seqlen % win
                tail = kv[-win:]
                self.window_kv[cutoff:win], self.window_kv[:cutoff] = tail[:win - cutoff], tail[win - cutoff:]
            window = kv
        else:
            self.window_kv[start_pos % win] = kv[0]
            window = self.window_kv
        idxs = window_idxs(win, seqlen, start_pos, x.device)
        if start_pos != 0:
            idxs = idxs.expand(seqlen, -1)
        keys = window

        if self.ratio:
            ratio = self.ratio
            compress_len = (start_pos + seqlen) // ratio
            latent = None
            if self.is_kv_source:
                latent = self.compressor(x, start_pos)
                shared.compress_kv = self.compress_kv
            offset = window.size(0)
            if not self.is_index_source:
                compress_idxs = shared.topk_idxs
            elif compress_len == 0:
                compress_idxs = torch.empty(seqlen, 0, dtype=torch.int32, device=x.device)
            else:
                compress_idxs = self.indexer(x, qr, latent, start_pos, offset, shared)
            if self.is_index_source:
                shared.topk_idxs = compress_idxs
            if latent is not None:
                latent = rotate_tail(latent, self.group_freqs(seqlen, start_pos), rd)
                latent = fp4_fake_quant(latent, 16, True)
                self.compress_kv[start_pos // ratio:start_pos // ratio + latent.size(0)] = latent
            keys = torch.cat([window, shared.compress_kv[:compress_len]], dim=0)
            idxs = torch.cat([idxs, compress_idxs], dim=-1)

        o = sparse_attention(q, keys, self.t("attn_sinks"), idxs, cfg.head_dim**-0.5)
        o = rotate_tail(o, freqs, rd, inverse=True)
        o = fp8_fake_quant(o.reshape(seqlen, cfg.o_groups, -1))
        wo_a = self.t("attn_output_a")  # [groups, rank, group_dim]
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        return linear(o.flatten(1), self.t("attn_output_b"))

    # -- MoE ------------------------------------------------------------
    def expert(self, x, gate_w, up_w, down_w, weight=None):
        limit = self.cfg.swiglu_limit
        gate = linear(x, gate_w)
        up = linear(x, up_w)
        if limit > 0:
            up = up.clamp(-limit, limit)
            gate = gate.clamp(max=limit)
        h = F.silu(gate) * up
        if weight is not None:
            h = weight * h
        return linear(h, down_w)

    def moe(self, x: torch.Tensor, routes: list):
        cfg = self.cfg
        scores = F.softplus(linear(x, self.t("ffn_gate_inp"))).sqrt()
        bias = self.w.f(self.p + "exp_probs_b.bias")
        indices = (scores + bias).topk(cfg.n_active, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * cfg.route_scale
        routes.append(indices.cpu().numpy().astype(np.int16))
        y = torch.zeros_like(x)
        used = sorted(set(indices.flatten().tolist()))
        for expert, (gate_w, up_w, down_w) in self.w.experts(self.id, used):
            rows, slot = torch.where(indices == expert)
            y[rows] += self.expert(x[rows], gate_w, up_w, down_w, weights[rows, slot, None])
        y += self.expert(x, self.t("ffn_gate_shexp"), self.t("ffn_up_shexp"), self.t("ffn_down_shexp"))
        return y

    # -- engram / hyper-connections ------------------------------------
    def engram(self, h: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        rows = hash_ids.flatten().tolist()
        unique = sorted(set(rows))
        table = self.w.rows(self.p + "engram_embd.weight", unique)
        lookup = {r: i for i, r in enumerate(unique)}
        embed = table[torch.tensor([lookup[r] for r in rows], device=h.device)].view(h.size(0), -1)
        kv = linear(embed, self.t("engram_wkv"))
        key, value = kv.split([cfg.hc_mult * cfg.dim, cfg.dim], dim=-1)
        key = key.unflatten(-1, (cfg.hc_mult, cfg.dim))
        weight = self.t("engram_q") * self.t("engram_k")
        eps = cfg.norm_eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * cfg.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.unsqueeze(-1) * value.unsqueeze(-2)

    def mixes(self, h, kind):
        cfg = self.cfg
        x = h.flatten(1)
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg.norm_eps)
        mixes = F.linear(x, self.t(f"hc_{kind}_fn")) * rsqrt
        return hc_split_sinkhorn(mixes, self.t(f"hc_{kind}_scale"), self.t(f"hc_{kind}_base"), cfg.hc_mult,
                                 cfg.hc_iters, cfg.hc_eps)

    def forward(self, h, start_pos, pre_mix, shared: Shared, routes: list):
        cfg = self.cfg
        residual = h
        attn_pre, attn_post, attn_comb = self.mixes(h, "attn")
        x = (pre_mix.unsqueeze(-1) * h).sum(1)
        x = rms_norm(x, self.t("attn_norm"), cfg.norm_eps)
        x = self.attention(x, start_pos, shared)
        h = attn_post.unsqueeze(-1) * x.unsqueeze(-2) + (attn_comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(1)

        residual = h
        ffn_pre, ffn_post, ffn_comb = self.mixes(h, "ffn")
        x = (attn_pre.unsqueeze(-1) * h).sum(1)
        x = rms_norm(x, self.t("ffn_norm"), cfg.norm_eps)
        x = self.moe(x, routes)
        h = ffn_post.unsqueeze(-1) * x.unsqueeze(-2) + (ffn_comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(1)
        return h, ffn_pre


class Transformer:
    def __init__(self, model: Model, device: torch.device, dtype: torch.dtype, threads: int, max_seq: int):
        self.cfg = Config.from_model(model)
        self.w = Weights(model, device, dtype, threads)
        self.engram_hash = EngramHash(self.cfg, model) if self.cfg.engram_layers else None
        self.layers = [Layer(self.cfg, self.w, i, max_seq) for i in range(self.cfg.n_layers)]
        self.shared = Shared()
        self.pos = 0

    def forward(self, tokens: list[int], dump=None, score=False):
        """Returns last-position logits and per-layer routes; ``dump(layer, h)`` sees each block output.
        With ``score``, also returns per-position (argmax, NLL of the next prompt token)."""
        cfg, w = self.cfg, self.w
        start_pos = self.pos
        hashes = self.engram_hash(tokens) if self.engram_hash else None
        h = w.rows("token_embd.weight", tokens)
        h = h.unsqueeze(1).repeat(1, cfg.hc_mult, 1)
        pre_mix = torch.zeros(len(tokens), cfg.hc_mult, device=h.device)
        pre_mix[:, 0] = 1.0
        routes: list = []
        for i, layer in enumerate(self.layers):
            if layer.has_engram:
                h = layer.engram(h, hashes[:, cfg.engram_layers.index(i)].to(h.device))
            h, pre_mix = layer.forward(h, start_pos, pre_mix, self.shared, routes)
            if dump is not None:
                dump(i, h)
        x = (pre_mix.unsqueeze(-1) * h).sum(1)
        scored = None
        if score:
            targets = torch.tensor(tokens[1:], device=x.device)
            argmax, nll = [], []
            for start in range(0, len(tokens) - 1, 256):
                part = rms_norm(x[start:start + 256][:len(targets) - start], w.f("output_norm.weight"), cfg.norm_eps)
                logp = torch.log_softmax(linear(part, w.get("output.weight")), -1)
                argmax.append(logp.argmax(-1).cpu())
                nll.append(-logp.gather(1, targets[start:start + part.size(0), None])[:, 0].cpu())
            scored = (torch.cat(argmax).numpy(), torch.cat(nll).numpy())
        x = rms_norm(x[-1:], w.f("output_norm.weight"), cfg.norm_eps)
        logits = linear(x, w.get("output.weight"))[0]
        self.pos += len(tokens)
        if score:
            return logits, np.stack(routes), scored
        return logits, np.stack(routes)


# --------------------------------------------------------------------------
# Engram layout verification (official engram.py algorithm)
# --------------------------------------------------------------------------
def official_layout(cfg: Config):
    from sympy import isprime

    seen, primes = set(), []
    for _ in cfg.engram_layers:
        row = []
        for _ in range(cfg.engram_ngram - 1):
            current = cfg.engram_vocab - 1
            for _ in range(cfg.engram_heads):
                current += 1
                while not isprime(current) or current in seen:
                    current += 1
                seen.add(current)
                row.append(current)
        primes.append(row)
    offsets = [np.cumsum([0, *row[:-1]]) for row in primes]
    bound = max(1, (np.iinfo(np.int64).max // cfg.engram_compressed_vocab) // 2)
    multipliers = []
    for layer in cfg.engram_layers:
        rng = np.random.default_rng(10007 * layer)
        multipliers.append(rng.integers(low=0, high=bound, size=(cfg.engram_ngram,), dtype=np.int64) * 2 + 1)
    return np.array(primes), np.array(offsets), np.array(multipliers)


def official_token_map(tokenizer_path: Path):
    from tokenizers import Regex, Tokenizer, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = Tokenizer.from_file(str(tokenizer_path))
    key_to_new: dict[str, int] = {}
    size = backend.get_vocab_size(with_added_tokens=True)
    lookup = [0] * size
    for token_id in range(size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return np.array(lookup), len(key_to_new)


def verify_engram(args) -> int:
    model = Model(args.model, args.engram)
    cfg = Config.from_model(model)
    primes, offsets, multipliers = official_layout(cfg)
    report = {
        "primes": bool(np.array_equal(primes, model["engram_primes.weight"].array())),
        "offsets": bool(np.array_equal(offsets, model["engram_offsets.weight"].array())),
        "multipliers": bool(np.array_equal(multipliers, model["engram_multipliers.weight"].array())),
        "table_rows": [int(primes[i].sum()) for i in range(len(cfg.engram_layers))],
        "gguf_rows": [model[f"blk.{layer}.engram_embd.weight"].shape[0] for layer in cfg.engram_layers],
    }
    report["rows"] = report["table_rows"] == report["gguf_rows"]
    if args.tokenizer:
        lookup, size = official_token_map(args.tokenizer)
        shipped = model["engram_token_map.weight"].array()
        report["token_map"] = bool(size == cfg.engram_compressed_vocab and np.array_equal(lookup, shipped))
        report["compressed_vocab"] = size
        from tokenizers import Tokenizer

        backend = Tokenizer.from_file(str(args.tokenizer))
        gguf_tokens = model.metadata_value("tokenizer.ggml.tokens")
        report["tokenizer_vocab"] = len(gguf_tokens) == backend.get_vocab_size(with_added_tokens=True) and all(
            backend.id_to_token(i) == t for i, t in enumerate(gguf_tokens))
    print(json.dumps(report, indent=2))
    checks = [v for k, v in report.items() if isinstance(v, bool)]
    return 0 if all(checks) else 1


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def encode(args) -> list[int]:
    if args.tokens:
        return [int(t) for t in args.tokens.split(",")]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text = args.text_file.read_text() if args.text_file else args.text
    ids = ([0] if args.bos else []) + tokenizer.encode(text, add_special_tokens=False).ids
    return ids[:args.max_tokens] if args.max_tokens else ids


def run(args) -> int:
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    model = Model(args.model, args.engram)
    tokens = encode(args)
    max_seq = len(tokens) + args.decode + 1
    net = Transformer(model, device, getattr(torch, args.weight_dtype), args.threads, max_seq)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    layers_dir = out / "layers"
    dump = None
    if args.dump_layers:
        layers_dir.mkdir(exist_ok=True)
        positions = sorted({p % len(tokens) for p in args.dump_positions}) if args.dump_positions else None

        def dump(layer, h):
            rows = h if positions is None or h.size(0) == 1 else h[positions]
            np.save(layers_dir / f"step{dump.step:03d}_layer{layer:02d}.npy", rows.cpu().numpy().astype(np.float32))
        dump.step = 0

    t0 = time.time()
    if args.score:
        logits, routes, (argmax, nll) = net.forward(tokens, dump, score=True)
        targets = np.array(tokens[1:])
        np.savez_compressed(out / "score.npz", argmax=argmax, nll=nll, targets=targets)
        score_report = {"positions": int(targets.size), "top1": float((argmax == targets).mean()),
                        "mean_nll": float(nll.mean()), "ppl": float(np.exp(nll.mean()))}
        for lo, hi in ((0, 512), (512, 1024), (1024, 2048), (2048, 1 << 30)):
            part = slice(lo, min(hi, targets.size))
            if lo < targets.size:
                score_report[f"top1_{lo}"] = float((argmax[part] == targets[part]).mean())
                score_report[f"nll_{lo}"] = float(nll[part].mean())
        print(json.dumps(score_report))
    else:
        logits, routes = net.forward(tokens, dump)
    timings = [time.time() - t0]
    all_logits = [logits.cpu().numpy()]
    all_routes = [routes]
    generated = []
    for step in range(args.decode):
        nxt = int(all_logits[-1].argmax())
        generated.append(nxt)
        if dump is not None:
            dump.step = step + 1
        t0 = time.time()
        logits, routes = net.forward([nxt], dump)
        timings.append(time.time() - t0)
        all_logits.append(logits.cpu().numpy())
        all_routes.append(routes)
    np.save(out / "logits.npy", np.stack(all_logits).astype(np.float32))
    np.savez_compressed(out / "routes.npz", *all_routes)
    manifest = {
        "model": str(Path(args.model).resolve()),
        "general.name": model.kv.get("general.name"),
        "prompt_tokens": tokens,
        "generated_tokens": generated,
        "device": args.device,
        "weight_dtype": args.weight_dtype,
        "seconds": timings,
        "reference": "official inference/model.py + engram.py, DeepSeek-V4.1-Flash-1017",
        "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.score:
        manifest["score"] = score_report
    if args.tokenizer:
        from tokenizers import Tokenizer

        manifest["generated_text"] = Tokenizer.from_file(str(args.tokenizer)).decode(generated)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("generated_tokens", "seconds")} |
                     ({"generated_text": manifest["generated_text"]} if "generated_text" in manifest else {}),
                     ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-engram", "run"):
        p = sub.add_parser(name)
        p.add_argument("--model", type=Path, required=True, help="first backbone shard")
        p.add_argument("--engram", type=Path, help="Engram GGUF (default: deepseek41.engram.file beside the model)")
        p.add_argument("--tokenizer", type=Path, help="official tokenizer.json")
    r = sub.choices["run"]
    prompt = r.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--text")
    prompt.add_argument("--tokens", help="comma-separated token ids")
    prompt.add_argument("--text-file", type=Path)
    r.add_argument("--bos", action="store_true", help="prepend <｜begin▁of▁sentence｜> to --text")
    r.add_argument("--decode", type=int, default=0, help="greedy decode steps after the prompt")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--weight-dtype", default="float32", choices=("float32", "bfloat16"))
    r.add_argument("--threads", type=int, default=16, help="expert dequantization threads")
    r.add_argument("--score", action="store_true", help="teacher-forced top-1 / NLL over the prompt")
    r.add_argument("--max-tokens", type=int, help="truncate the encoded prompt")
    r.add_argument("--dump-layers", action="store_true", help="save every block output per step")
    r.add_argument("--dump-positions", type=int, nargs="*", help="prefill positions to keep (negative ok)")
    args = parser.parse_args()
    return verify_engram(args) if args.command == "verify-engram" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
