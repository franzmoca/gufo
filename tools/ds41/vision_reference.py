#!/usr/bin/env python3
"""CPU reference for the DeepSeek V4.1 Flash vision path.

A torch port of the official inference/image_processor.py (plan_image_grid,
load_image), vision.py (ViT, Aligner) and Transformer.merge_image_embeddings,
in the official BF16 default dtype, reading the weights from the mmproj.
Writes the padded RGB8 pixels and the
image span's input embeddings, [IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE])
* h + [IMAGE_END], as FP32 rows; with --compare, reports how closely a GPU
span (ds41_vision_probe --output) matches.

    python3 -I tools/ds41/vision_reference.py --mmproj MMPROJ.gguf \\
        --image IMAGE --out DIR [--compare SPAN.f32]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gufo.gguf import parse_gguf  # noqa: E402

PATCH = 14
DOWNSAMPLE = 3
MAX_TOKENS = 1024
MIN_PIXELS = 544 * 544
DIM = 1024
HEADS = 16
INTER = 2816
LAYERS = 32
THETA = 10000.0


# image_processor.py
def llm_grid(best_height, best_width):
    return math.ceil((best_height // PATCH) / DOWNSAMPLE), math.ceil((best_width // PATCH) / DOWNSAMPLE)


def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def solve_resize_ratio(height, width):
    r = height / width
    max_w_float = math.sqrt((MAX_TOKENS - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = PATCH * DOWNSAMPLE
    if max_w_float < 1.0:
        return (MAX_TOKENS - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (MAX_TOKENS - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / PATCH) * PATCH, math.floor(width * beta / PATCH) * PATCH


def plan_image_grid(width, height):
    if 0 < width * height < MIN_PIXELS:
        ratio = (MIN_PIXELS / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / PATCH) * PATCH
    best_height = math.ceil(height / PATCH) * PATCH
    n_llm_h, n_llm_w = llm_grid(best_height, best_width)
    if num_image_tokens(n_llm_h, n_llm_w) > MAX_TOKENS:
        best_height, best_width = solve_resize_ratio(height, width)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width)
    return n_llm_h, n_llm_w, best_height, best_width


def load_image(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height)
    image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = np.asarray(image, dtype=np.uint8)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    n_vit_h, n_vit_w = best_height // PATCH, best_width // PATCH
    patches = x.reshape(3, n_vit_h, PATCH, n_vit_w, PATCH).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, PATCH, PATCH)
    return pixels, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


# vision.py
def vision_cos_sin(n_h, n_w, dim):
    inv_freq = 1.0 / (THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x, cos, sin):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def rms_norm(x, weight, eps=1e-6):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


class Weights:
    def __init__(self, path, dtype=torch.bfloat16):
        self.dtype = dtype
        self.state = parse_gguf(path)
        self.raw = Path(path).read_bytes()
        self.tensors = {t["name"]: t for t in self.state["tensors"]}

    def __call__(self, name, dtype=None):
        t = self.tensors[name]
        count = int(np.prod(t["shape"]))
        at = self.state["data_offset"] + t["offset"]
        if t["type"] == 0:
            array = np.frombuffer(self.raw, dtype="<f4", count=count, offset=at)
        elif t["type"] == 30:
            array = (np.frombuffer(self.raw, dtype="<u2", count=count, offset=at).astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"{name} has unexpected type {t['type_name']}")
        return torch.from_numpy(array.copy().reshape(t["shape"])).to(dtype or self.dtype)


def encode(w, patches, n_h, n_w):
    """ViT and Aligner forward, as vision.py."""
    n = patches.size(0)
    x = F.linear(patches.flatten(1), w("v.patch_embd.weight").reshape(DIM, -1), w("v.patch_embd.bias"))
    cos, sin = vision_cos_sin(n_h, n_w, DIM // HEADS // 2, )
    head_dim = DIM // HEADS
    for layer in range(LAYERS):
        p = f"v.blk.{layer}."
        h = rms_norm(x, w(p + "ln1.weight", torch.float32))
        q, k, v = (t.view(n, HEADS, head_dim)
                   for t in F.linear(h, w(p + "attn_qkv.weight"), w(p + "attn_qkv.bias")).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        x = x + F.linear(o.transpose(0, 1).reshape(n, -1), w(p + "attn_out.weight"), w(p + "attn_out.bias"))
        h = rms_norm(x, w(p + "ln2.weight", torch.float32))
        gate = F.linear(h, w(p + "ffn_gate.weight"))
        up = F.linear(h, w(p + "ffn_up.weight"))
        x = x + F.linear(F.silu(gate) * up, w(p + "ffn_down.weight"))
    x = rms_norm(x, w("v.post_ln.weight", torch.float32))
    r = DOWNSAMPLE
    x = x.view(n_h, n_w, -1).permute(2, 0, 1)
    x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
    x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
    x = F.linear(x, w("mm.1.weight"), w("mm.1.bias"))
    return F.linear(F.gelu(x), w("mm.2.weight"), w("mm.2.bias"))


def span(w, embeds, n_llm_h, n_llm_w):
    """merge_image_embeddings over one image's span, in reading order."""
    start = w("v.token_embd.img_start")
    end = w("v.token_embd.img_end")
    newline = w("v.image_newline")
    rows = [start]
    for y in range(n_llm_h):
        rows.extend(embeds[y * n_llm_w:(y + 1) * n_llm_w])
        rows.append(newline)
    rows.append(end)
    return torch.stack(rows).float().numpy()


def compare(reference, candidate):
    diff = candidate.astype(np.float64) - reference
    rel = math.sqrt((diff ** 2).sum() / max((reference.astype(np.float64) ** 2).sum(), 1e-30))
    dots = (candidate * reference).sum(1)
    norms = np.linalg.norm(candidate, axis=1) * np.linalg.norm(reference, axis=1)
    cosine = (dots / np.maximum(norms, 1e-30)).min()
    return {"relative_rms": rel, "max_abs": float(np.abs(diff).max()), "worst_row_cosine": float(cosine)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="directory for pixels.rgb, span.npy, grid.json")
    parser.add_argument("--compare", type=Path, help="GPU span rows (FP32) to compare")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fp32", action="store_true",
                        help="run in FP32 instead of the official BF16, as a precision yardstick")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.set_default_dtype(dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    pixels, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(args.image)
    patches = patches.to(dtype)
    grid = {"width": pixels.shape[1], "height": pixels.shape[0], "vit_width": n_vit_w, "vit_height": n_vit_h,
            "llm_width": n_llm_w, "llm_height": n_llm_h, "tokens": num_image_tokens(n_llm_h, n_llm_w)}
    (args.out / "grid.json").write_text(json.dumps(grid) + "\n")
    (args.out / "pixels.rgb").write_bytes(pixels.tobytes())
    w = Weights(args.mmproj, dtype)
    with torch.inference_mode():
        rows = span(w, encode(w, patches, n_vit_h, n_vit_w), n_llm_h, n_llm_w)
    np.save(args.out / ("span_fp32.npy" if args.fp32 else "span.npy"), rows)
    print(json.dumps(grid))
    if args.compare:
        candidate = np.fromfile(args.compare, dtype="<f4")
        if candidate.size != rows.size:
            print(f"GPU span has {candidate.size} values, reference {rows.size}")
            return 1
        print(json.dumps(compare(rows, candidate.reshape(rows.shape))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
