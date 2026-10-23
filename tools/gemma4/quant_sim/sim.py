"""Gemma 4 31B weight/activation quantization simulation against BF16.

Streams the BF16 checkpoint one decoder layer at a time and carries every
requested variant's hidden states through it, so one pass over the weights
scores all variants. Projections run on binary16 inputs with FP32 sums, as
Gufo's prefill does; norms, attention and the residual stay FP32. A variant
replaces the projection weights (and optionally quantizes their inputs); the
embedding and the tied LM head stay BF16 in every variant.

Scores are per evaluation token against the `bf16` variant: KL(ref || q),
top-1 agreement and the next-token NLL.

    python -I tools/gemma4/quant_sim/sim.py --model DIR --tokens FILE \
        --variants bf16,q4kxl,w4 --out results.json
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastops  # noqa: E402
import quant  # noqa: E402

PREFIX = "model.language_model."
CLASSES = ("qkv", "o", "gate_up", "down")
LINEAR_CLASS = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o",
                "gate": "gate_up", "up": "gate_up", "down": "down"}
HF_NAME = {"q": "self_attn.q_proj", "k": "self_attn.k_proj",
           "v": "self_attn.v_proj", "o": "self_attn.o_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj",
           "down": "mlp.down_proj"}


class Checkpoint:
    def __init__(self, directory):
        index = json.loads((directory / "model.safetensors.index.json")
                           .read_text())["weight_map"]
        self.files = {}
        self.where = {}
        for name, file in index.items():
            if file not in self.files:
                self.files[file] = safe_open(str(directory / file), "pt")
            self.where[name] = self.files[file]

    def get(self, name):
        return self.where[PREFIX + name].get_tensor(PREFIX + name)

    def has(self, name):
        return PREFIX + name in self.where


class Config:
    def __init__(self, directory):
        c = json.loads((directory / "config.json").read_text())["text_config"]
        self.hidden = c["hidden_size"]
        self.layers = c["num_hidden_layers"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.global_kv_heads = c["num_global_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.global_head_dim = c["global_head_dim"]
        self.window = c["sliding_window"]
        self.eps = c["rms_norm_eps"]
        self.softcap = c["final_logit_softcapping"]
        self.sliding = [t == "sliding_attention" for t in c["layer_types"]]
        rope = c["rope_parameters"]
        self.theta_local = rope["sliding_attention"]["rope_theta"]
        self.theta_global = rope["full_attention"]["rope_theta"]
        self.partial_global = rope["full_attention"]["partial_rotary_factor"]


def rms_norm(x, w, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def rope_tables(cfg, length, device):
    pos = torch.arange(length, device=device, dtype=torch.float64)

    def table(dim, theta, rotated_pairs):
        i = torch.arange(dim // 2, device=device, dtype=torch.float64)
        inv = theta ** (-2.0 * i / dim)
        inv[rotated_pairs:] = 0.0  # proportional RoPE: the rest stay put
        ang = pos[:, None] * inv[None, :]
        return ang.cos().float(), ang.sin().float()

    local = table(cfg.head_dim, cfg.theta_local, cfg.head_dim // 2)
    glob = table(cfg.global_head_dim, cfg.theta_global,
                 int(cfg.partial_global * cfg.global_head_dim // 2))
    return local, glob


def apply_rope(x, cos, sin):
    # x [T, H, D], NEOX halves.
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1)


class Layer:
    """One decoder layer's BF16 tensors on the device."""

    def __init__(self, ckpt, cfg, index, device):
        p = f"layers.{index}."
        self.sliding = cfg.sliding[index]
        self.w = {}
        for name, hf in HF_NAME.items():
            if ckpt.has(p + hf + ".weight"):
                self.w[name] = ckpt.get(p + hf + ".weight").to(device)
        self.w16 = {n: w.half() for n, w in self.w.items()}
        def vec(n):
            return ckpt.get(p + n).to(device).float()
        self.attn_norm = vec("input_layernorm.weight")
        self.post_attn_norm = vec("post_attention_layernorm.weight")
        self.ffn_norm = vec("pre_feedforward_layernorm.weight")
        self.post_ffn_norm = vec("post_feedforward_layernorm.weight")
        self.q_norm = vec("self_attn.q_norm.weight")
        self.k_norm = vec("self_attn.k_norm.weight")
        self.scalar = vec("layer_scalar").item()
        self.head_dim = cfg.head_dim if self.sliding else cfg.global_head_dim
        self.kv_heads = cfg.kv_heads if self.sliding else cfg.global_kv_heads


ATTENTION_HALF = True


def attention(q, k, v, sliding, window, chunk=1024):
    """q [T, H, D], k/v [T, KV, D]; scale 1, causal, sliding window incl.
    self. Queries go in chunks so long prompts keep the scores bounded."""
    t = q.shape[0]
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1).transpose(0, 1)
    v = v.repeat_interleave(group, dim=1).transpose(0, 1)
    q = q.transpose(0, 1)
    out = []
    for q0 in range(0, t, chunk):
        q1 = min(q0 + chunk, t)
        k0 = max(0, q0 - window + 1) if sliding else 0
        qi = torch.arange(q0, q1, device=q.device)
        ki = torch.arange(k0, q1, device=q.device)
        mask = ki[None, :] <= qi[:, None]
        if sliding:
            mask &= ki[None, :] > qi[:, None] - window
        if ATTENTION_HALF:
            # binary16 products with FP32 sums, FP32 softmax.
            sc = torch.matmul(q[:, q0:q1].half(),
                              k[:, k0:q1].half().transpose(1, 2)).float()
            sc = sc.masked_fill(~mask, float("-inf")).softmax(-1)
            out.append(torch.matmul(sc.half(), v[:, k0:q1].half()).float())
        else:
            out.append(F.scaled_dot_product_attention(
                q[:, q0:q1], k[:, k0:q1], v[:, k0:q1], attn_mask=mask,
                scale=1.0))
    return torch.cat(out, dim=1).transpose(0, 1).reshape(t, -1)


def block(x, layer, lin, cfg, rope, hooks=None):
    """One decoder layer over one sequence x [T, hidden] (FP32).

    `lin(name, input)` evaluates a projection. `hooks(cls, input)`, when set,
    sees each projection class's input before it is used (calibration)."""
    t = x.shape[0]
    d = layer.head_dim
    cos, sin = rope[0] if layer.sliding else rope[1]
    cos, sin = cos[:t], sin[:t]
    h = rms_norm(x, layer.attn_norm, cfg.eps)
    if hooks:
        hooks("qkv", h)
    q = lin("q", h).view(t, cfg.heads, d)
    k = lin("k", h).view(t, layer.kv_heads, d)
    v = lin("v", h).view(t, layer.kv_heads, d) if "v" in layer.w else k
    q = apply_rope(rms_norm(q, layer.q_norm, cfg.eps), cos, sin)
    v = rms_norm(v, None, cfg.eps)
    k = apply_rope(rms_norm(k, layer.k_norm, cfg.eps), cos, sin)
    a = attention(q, k, v, layer.sliding, cfg.window)
    if hooks:
        hooks("o", a)
    x = x + rms_norm(lin("o", a), layer.post_attn_norm, cfg.eps)
    h = rms_norm(x, layer.ffn_norm, cfg.eps)
    if hooks:
        hooks("gate_up", h)
    g = F.gelu(lin("gate", h), approximate="tanh") * lin("up", h)
    if hooks:
        hooks("down", g)
    f = rms_norm(lin("down", g), layer.post_ffn_norm, cfg.eps)
    return (x + f) * layer.scalar


def pack_hessian(h):
    """Exact FP32: the down and o Hessians are close to singular, and a
    binary16 correlation copy (4e-4 relative) made them fail Cholesky at the
    1% GPTQ damping that the exact matrices pass."""
    return h.cpu()


def unpack_hessian(packed, dev):
    return packed.to(dev)


def half_matmul(x, w16):
    """binary16 inputs, FP32 sums; FP32 fallback where binary16 overflows."""
    y = torch.mm(x.half(), w16.t()).float()
    if not torch.isfinite(y).all():
        y = torch.mm(x, w16.float().t())
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gguf", type=Path, action="append", default=[],
                   help="name=path of a GGUF for gguf:<name> variants")
    p.add_argument("--layers", type=int, default=0,
                   help="debug: run only the first N layers")
    p.add_argument("--calib-rows", default="",
                   help="calibration rows per set, e.g. std=32,chat=32 "
                        "(unlisted sets: all rows)")
    p.add_argument("--hessian-precision", choices=["kernel", "half", "fp32"],
                   default="kernel",
                   help="kernel: binary16 WMMA products accumulated in FP32 "
                        "(kernels.hip); half: binary16 products rounded per "
                        "row; fp32: FP32 products")
    p.add_argument("--fp32-ref", action="store_true",
                   help="add a `fp32` variant with FP32 projections")
    p.add_argument("--resume", action="store_true",
                   help="continue from the checkpoint next to --out")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--drop-eval", default="",
                   help="comma-separated evaluation items to leave out")
    p.add_argument("--hessian-cache", type=Path,
                   help="directory keeping per-layer calibration Hessians")
    p.add_argument("--attention", choices=["half", "fp32"], default="half",
                   help="binary16 attention products (as Gufo's prefill) or "
                        "FP32")
    args = p.parse_args()
    global ATTENTION_HALF
    ATTENTION_HALF = args.attention == "half"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    cfg = Config(args.model)
    ckpt = Checkpoint(args.model)
    data = torch.load(args.tokens)
    # Evaluation sequences may differ in length; only positions from
    # eval_score_from on (the model's own reply) are scored.
    eval_ids = list(data["eval"])
    score_from = data.get("eval_score_from", [1] * len(eval_ids))
    meta = data["eval_meta"]
    drop = set(filter(None, args.drop_eval.split(",")))
    keep = [i for i, (_, name) in enumerate(meta) if name not in drop]
    assert len(keep) == len(meta) - len(drop), "unknown --drop-eval item"
    eval_ids = [eval_ids[i] for i in keep]
    score_from = [score_from[i] for i in keep]
    meta = [meta[i] for i in keep]
    calib_rows = {"std": data["calib"]}
    calib_rows.update({k[6:]: v for k, v in data.items()
                       if k.startswith("calib_")})
    for item in filter(None, args.calib_rows.split(",")):
        name, count = item.split("=")
        calib_rows[name] = calib_rows[name][:int(count)]
    ggufs = dict(s.split("=", 1) for s in map(str, args.gguf))
    names = args.variants.split(",")
    if args.fp32_ref:
        names.append("fp32")
    if "bf16" not in names:
        names.insert(0, "bf16")
    variants = {n: quant.make_variant(n, ggufs) for n in names}
    sets = sorted(set().union(*(quant.calib_sets(v) for v in variants.values())))
    # Hessians come from the disk cache when every layer of a set is there.
    cache_dirs = {}
    if args.hessian_cache:
        for cs in sets:
            tag = hashlib.sha1(calib_rows[cs].numpy().tobytes()).hexdigest()[:12]
            # A complete cache of either precision is reused; new sets are
            # computed at --hessian-precision.
            prec = {"fp32": "-f32", "half": "-h16",
                    "kernel": "-wmma"}[args.hessian_precision]
            cache_dirs[cs] = args.hessian_cache / f"{cs}-{tag}{prec}"
            for suffix in ("-f32", "-wmma"):
                d = args.hessian_cache / f"{cs}-{tag}{suffix}"
                if all((d / f"layer{li}.pt").exists()
                       for li in range(args.layers or cfg.layers)):
                    cache_dirs[cs] = d
            cache_dirs[cs].mkdir(parents=True, exist_ok=True)
    cached = {cs for cs in sets if cs in cache_dirs and all(
        (cache_dirs[cs] / f"layer{li}.pt").exists()
        for li in range(args.layers or cfg.layers))}
    for cs in sets:
        print(f"calibration {cs}: {len(calib_rows[cs])} rows, "
              f"{'cached' if cs in cached else 'computed'}", flush=True)

    embed = ckpt.get("embed_tokens.weight").to(dev)
    embed_scale = torch.tensor(cfg.hidden ** 0.5,
                               dtype=torch.bfloat16).float().item()

    def embed_rows(ids):
        return [embed[r.to(dev)].float() * embed_scale for r in ids]

    states = {n: embed_rows(eval_ids) for n in names}
    # Calibration rows live in host memory and go to the device one row at a
    # time (v6 alone is ~5.4 GB of FP32 states).
    calib_states = {cs: [x.cpu() for x in embed_rows(calib_rows[cs])]
                    for cs in sets
                    if cs not in cached}
    length = max(len(x) for x in eval_ids)
    rope = rope_tables(cfg, max([length] + [calib_rows[cs].shape[1]
                                            for cs in sets]), dev)
    layers = args.layers or cfg.layers
    report = {n: {"bits": {}} for n in names}
    ckpt_path = args.out.with_suffix(".ckpt.pt")
    first = 0
    if args.resume and ckpt_path.exists():
        saved = torch.load(ckpt_path)
        assert saved["names"] == names, "checkpoint has other variants"
        first = saved["layer"]
        states = {n: [x.to(dev) for x in v] for n, v in saved["states"].items()}
        calib_states = {cs: list(v)
                        for cs, v in saved["calib"].items()}
        report = saved["report"]
        del saved
        print(f"resumed after layer {first}", flush=True)
    t0 = time.time()
    for li in range(first, layers):
        layer = Layer(ckpt, cfg, li, dev)
        hessians = {}
        for cs in sets:
            path = cache_dirs[cs] / f"layer{li}.pt" if cs in cache_dirs else None
            if cs in cached:
                hessians[cs] = {c: unpack_hessian(v, dev)
                                for c, v in torch.load(path).items()}
                continue
            # Input Hessians of the BF16 model's calibration rows, shared by
            # every variant calibrated on this set, in FP32.
            sums = {c: None for c in CLASSES}

            def collect(c, inp, sums=sums):
                if args.hessian_precision == "kernel":
                    if sums[c] is None:
                        k = inp.shape[1]
                        sums[c] = torch.zeros(k, k, device=inp.device)
                    fastops.hessian_accumulate(sums[c], inp)
                    return
                if args.hessian_precision == "fp32":
                    if sums[c] is None:
                        sums[c] = inp.t() @ inp
                    else:
                        sums[c].addmm_(inp.t(), inp)
                    return
                # Columns scaled to max |x| = 1 so the binary16 products
                # cannot overflow; FP32 rescale and sum. Normalized error
                # ~4e-4 against FP32 products: enough to break the nearly
                # singular down/o Hessians, so not the default.
                col = inp.abs().amax(0).clamp_min(1e-30)
                xs = (inp / col).half()
                hh = (xs.t() @ xs).float().mul_(col[:, None]).mul_(col[None, :])
                if sums[c] is None:
                    sums[c] = hh
                else:
                    sums[c].add_(hh)

            def lin_ref(name, inp):
                return half_matmul(inp, layer.w16[name])

            calib_states[cs] = [block(r.to(dev), layer, lin_ref, cfg, rope,
                                      hooks=collect).cpu()
                                for r in calib_states[cs]]
            rows = sum(r.shape[0] for r in calib_states[cs])
            if args.hessian_precision == "kernel":
                sums = {c: fastops.symmetrize(h) for c, h in sums.items()}
            hessians[cs] = {c: h / rows for c, h in sums.items()}
            del sums
            if path is not None:
                torch.save({c: pack_hessian(h) for c, h in hessians[cs].items()},
                           path)
        torch.cuda.empty_cache()
        for n in names:
            var = variants[n]
            prepared = {}
            for cls in CLASSES:
                prepared[cls] = var.prepare(li, cls, layer, hessians)
            for cls, prep in prepared.items():
                report[n]["bits"].setdefault(cls, [0, 0])
                b, c = prep.bits()
                report[n]["bits"][cls][0] += b
                report[n]["bits"][cls][1] += c

            def lin(name, inp, prepared=prepared):
                return prepared[LINEAR_CLASS[name]].apply(name, inp, li)

            out = []
            for si, x in enumerate(states[n]):
                quant.SPLIT = score_from[si]
                out.append(block(x, layer, lin, cfg, rope))
            quant.SPLIT = None
            states[n] = out
            del prepared
        del layer
        torch.cuda.empty_cache()
        print(f"layer {li + 1}/{layers} {time.time() - t0:.0f}s", flush=True)
        if (li + 1) % args.checkpoint_every == 0 and li + 1 < layers:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"names": names, "layer": li + 1,
                        "states": {n: [x.cpu() for x in v]
                                   for n, v in states.items()},
                        "calib": {cs: [x.cpu() for x in v]
                                  for cs, v in calib_states.items()},
                        "report": report}, ckpt_path)

    # Final norm, tied LM head (BF16, FP32 sums), softcap; scores vs bf16.
    norm = ckpt.get("norm.weight").to(dev).float()
    head = embed.half()
    def logprobs(x):
        h = rms_norm(x, norm, cfg.eps)
        z = torch.mm(h.half(), head.t()).float()
        if cfg.softcap:
            z = cfg.softcap * torch.tanh(z / cfg.softcap)
        return F.log_softmax(z, dim=-1)

    per_token = {n: {"kl": [], "agree": [], "nll": []} for n in names}
    token_cat = []
    chunk = 256
    for si, ids in enumerate(eval_ids):
        labels = ids[1:].to(dev)
        first = score_from[si] - 1  # position predicting the first reply token
        for c0 in range(first, len(ids) - 1, chunk):
            c1 = min(c0 + chunk, len(ids) - 1)
            token_cat += [meta[si][0]] * (c1 - c0)
            ref = logprobs(states["bf16"][si][c0:c1])
            ref_top = ref.argmax(-1)
            for n in names:
                lp = ref if n == "bf16" else logprobs(states[n][si][c0:c1])
                kl = (ref.exp() * (ref - lp)).sum(-1)
                per_token[n]["kl"].append(kl.cpu())
                per_token[n]["agree"].append((lp.argmax(-1) == ref_top).cpu())
                per_token[n]["nll"].append(
                    -lp.gather(1, labels[c0:c1, None])[:, 0].cpu())

    token_seq = []
    for si, ids in enumerate(eval_ids):
        token_seq += [si] * (len(ids) - score_from[si])
    token_seq = torch.tensor(token_seq)
    seq_count = len(eval_ids)
    cats = sorted(set(token_cat))
    cat_index = {c: torch.tensor([i for i, x in enumerate(token_cat) if x == c])
                 for c in cats}
    for n in names:
        kl = torch.cat(per_token[n]["kl"]).double()
        agree = torch.cat(per_token[n]["agree"]).double()
        nll = torch.cat(per_token[n]["nll"]).double()
        r = report[n]
        r["kl_mean"] = kl.mean().item()
        # 95% interval from resampling whole sequences (tokens within one
        # reply are correlated).
        seq_sum = torch.zeros(seq_count, dtype=torch.float64).index_add_(
            0, token_seq, kl)
        seq_n = torch.zeros(seq_count, dtype=torch.float64).index_add_(
            0, token_seq, torch.ones_like(kl))
        g = torch.Generator().manual_seed(0)
        pick = torch.randint(0, seq_count, (2000, seq_count), generator=g)
        boot = seq_sum[pick].sum(1) / seq_n[pick].sum(1)
        r["kl_ci95"] = [boot.quantile(0.025).item(), boot.quantile(0.975).item()]
        r["kl_by_sequence"] = (seq_sum / seq_n.clamp_min(1)).tolist()
        r["kl_p99"] = kl.quantile(0.99).item()
        r["top1"] = agree.mean().item()
        r["ppl"] = math.exp(nll.mean().item())
        r["by_category"] = {}
        for cat in cats:
            idx = cat_index[cat]
            r["by_category"][cat] = {
                "kl_mean": kl[idx].mean().item(),
                "top1": agree[idx].mean().item(),
                "ppl": math.exp(nll[idx].mean().item())}
        total_b = sum(b for b, _ in r["bits"].values())
        total_c = sum(c for _, c in r["bits"].values())
        r["bpw"] = total_b / total_c if total_c else None
        r["bpw_by_class"] = {k: b / c for k, (b, c) in r["bits"].items()}
        del r["bits"]
        r["spec"] = variants[n].describe()
        print(f"{n:28s} bpw {r['bpw'] or 0:5.2f}  KL {r['kl_mean']:.5f} "
              f"[{r['kl_ci95'][0]:.4f}, {r['kl_ci95'][1]:.4f}] "
              f"p99 {r['kl_p99']:.4f}  top1 {r['top1']:.4f}  "
              f"ppl {r['ppl']:.3f}  " + "  ".join(
                  f"{c}:{v['kl_mean']:.4f}"
                  for c, v in r["by_category"].items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"layers": layers, "eval_tokens": len(token_cat),
         "dropped": sorted(drop), "attention": args.attention,
         "calib_rows": {k: len(v) for k, v in calib_rows.items()},
         "eval_items": meta, "hessian_precision": args.hessian_precision,
         "variants": report}, indent=1))


if __name__ == "__main__":
    main()
