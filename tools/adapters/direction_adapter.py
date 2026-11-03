#!/usr/bin/env python3
"""Converts steering and abliteration files to direction adapters.

A direction adapter (src/core/direction_adapter.hpp) is a GGUF with
general.type = adapter, adapter.type = direction, adapter.direction.mode
("project" or "add"), an optional adapter.direction.scale and F32 tensors
blk.{layer}.{site}.direction of ne = [width, k].

  from-abl   mako's MAKOABL1 files (project mode; the vectors and site names
             are copied unchanged, all-zero sets dropped).
  from-lora  rank-1 abliteration LoRAs whose lora_b is one direction r for
             every matrix. Each residual writer W' = W - r (r^T W) becomes
             the projection of r at the sublayer output, which is exact on
             any quantization of the base model; the stored lora_a halves
             (r^T W of the quantization the LoRA was built from) are dropped.
             --model checks lora_a against -r^T W of that model's attention
             output projection.
  inspect    prints a direction adapter's metadata and sets.
"""
from __future__ import annotations

import argparse
import mmap
from pathlib import Path
import re
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.gguf import parse_gguf, write_gguf  # noqa: E402

# LoRA target matrices -> the sublayer output site each writes into.
# A layer whose shared and routed down projections carry the same direction
# projects the whole FFN output (ffn_out).
LORA_SITES = {
    "deepseek41": {
        "attn_output_b": "attn_out",
        "ffn_down_shexp": "shexp_out",
        "ffn_down_exps": "routed_out",
    },
}
WIDTHS = {"deepseek41": 5120}


def read_abl(path: Path):
    data = path.read_bytes()
    if data[:8] != b"MAKOABL1":
        raise SystemExit(f"{path} is not a MAKOABL1 file")
    width, layers, n_site, k = struct.unpack_from("<4I", data, 8)
    off = 24
    sites = [data[off + 16 * i:off + 16 * (i + 1)].rstrip(b"\0").decode() for i in range(n_site)]
    off += 16 * n_site
    count = layers * n_site * k * width
    if len(data) != off + 4 * count:
        raise SystemExit(f"{path}: {len(data)} bytes, expected {off + 4 * count}")
    vectors = np.frombuffer(data, "<f4", count, off).reshape(layers, n_site, k, width)
    return width, sites, vectors


def from_abl(args):
    width, sites, vectors = read_abl(args.input)
    if args.arch in WIDTHS and WIDTHS[args.arch] != width:
        raise SystemExit(f"{args.input}: width {width}, {args.arch} has {WIDTHS[args.arch]}")
    tensors = []
    for layer in range(vectors.shape[0]):
        for s, site in enumerate(sites):
            v = vectors[layer, s]
            if np.any(v != 0):
                tensors.append((f"blk.{layer}.{site}.direction", v))
    metadata = [
        ("general.architecture", args.arch),
        ("general.type", "adapter"),
        ("general.name", args.name or args.input.stem),
        ("adapter.type", "direction"),
        ("adapter.direction.mode", "project"),
        ("adapter.direction.scale", 1.0),
        ("adapter.direction.source", f"MAKOABL1 {args.input.name}"),
    ]
    write_gguf(args.output, metadata, tensors)
    print(f"{args.output}: {len(tensors)} sets from {len(sites)} sites {sites}, k={vectors.shape[2]}")


class Gguf:
    def __init__(self, path: Path):
        self.info = parse_gguf(path)
        self.kv = {**self.info["metadata"], **self.info["user_kv"]}
        self._file = path.open("rb")
        self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors = {t["name"]: t for t in self.info["tensors"]}

    def f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        if t["type"] != 0:
            raise SystemExit(f"{name}: type {t['type_name']}, expected F32")
        count = int(np.prod(t["shape"]))
        return np.frombuffer(self.data, "<f4", count, self.info["data_offset"] + t["offset"]).reshape(t["shape"])


def check_against_model(model_path: Path, layer: int, r: np.ndarray, lora_a: np.ndarray, scale: float):
    """Compares lora_a with -scale r^T W for one attention output projection."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ds41"))
    from gguf_model import Model  # noqa: PLC0415

    w = Model(model_path)[f"blk.{layer}.attn_output_b.weight"].array()  # [out, in]
    expected = -scale * (r.astype(np.float64) @ w.astype(np.float64))
    a = lora_a.astype(np.float64).ravel()
    cos = float(a @ expected / (np.linalg.norm(a) * np.linalg.norm(expected)))
    rel = float(np.linalg.norm(a - expected) / np.linalg.norm(expected))
    print(f"layer {layer} attn_output_b: cos(lora_a, -r^T W) {cos:+.6f}, relative difference {rel:.4f}")
    if cos < 0.9:
        raise SystemExit("lora_a is not -r^T W on this model: the LoRA is not a projection of r")


def from_lora(args):
    src = Gguf(args.input)
    arch = src.kv.get("general.architecture")
    if src.kv.get("general.type") != "adapter" or src.kv.get("adapter.type") != "lora":
        raise SystemExit(f"{args.input} is not a LoRA adapter")
    if arch not in LORA_SITES:
        raise SystemExit(f"no site table for architecture {arch!r}")
    sites, width = LORA_SITES[arch], WIDTHS[arch]
    pattern = re.compile(r"blk\.(\d+)\.(\w+)\.weight\.lora_([ab])$")
    per_layer: dict[int, dict[str, np.ndarray]] = {}
    lora_a = {}
    r = None
    for name in src.tensors:
        m = pattern.match(name)
        if not m:
            raise SystemExit(f"unexpected tensor {name}")
        layer, matrix, half = int(m.group(1)), m.group(2), m.group(3)
        if matrix not in sites:
            raise SystemExit(f"{name}: {matrix} has no site in the {arch} table")
        if half == "a":
            lora_a[(layer, matrix)] = src.f32(name)
            continue
        b = src.f32(name).reshape(-1, width)  # every expert's (or the one) column
        if np.any(b != b[0]):
            raise SystemExit(f"{name}: lora_b differs between experts; not a rank-1 projection LoRA")
        if r is not None and not np.array_equal(r, b[0]):
            raise SystemExit(f"{name}: lora_b differs from the other matrices; not one direction")
        r = b[0].copy()
        per_layer.setdefault(layer, {})[matrix] = b[0]
    if r is None:
        raise SystemExit("no lora_b tensors")
    for layer, matrix in lora_a:
        if matrix not in per_layer.get(layer, {}):
            raise SystemExit(f"blk.{layer}.{matrix}: lora_a without lora_b")
    alpha = float(src.kv.get("adapter.lora.alpha", 1.0))
    scale = alpha  # rank 1: alpha / rank
    norm = float(np.linalg.norm(r))
    print(f"{args.input.name}: one direction r (|r| = {norm:.6f}) in {sum(len(v) for v in per_layer.values())} "
          f"matrices over {len(per_layer)} layers, LoRA scale {scale}")
    if args.model:
        layer = int(src.kv.get("abliterate.direction_layer", min(per_layer)))
        check_against_model(args.model, layer, r, lora_a[(layer, "attn_output_b")], scale)
    else:
        print("warning: without --model the LoRA's sign is assumed to remove r (lora_a = -r^T W)")
    # W' = W - scale r r^T W adds -scale (r . y) r to y = W x: the projection
    # y -= (y . v) v with v = sqrt(scale) r.
    if not scale > 0:
        raise SystemExit(f"LoRA scale {scale} does not describe a removal")
    v = (np.sqrt(scale) * r).astype(np.float32)
    tensors = []
    for layer in sorted(per_layer):
        matrices = per_layer[layer]
        if "attn_output_b" in matrices:
            tensors.append((f"blk.{layer}.attn_out.direction", v[None, :]))
        shexp, routed = "ffn_down_shexp" in matrices, "ffn_down_exps" in matrices
        if shexp and routed:
            tensors.append((f"blk.{layer}.ffn_out.direction", v[None, :]))
        elif shexp:
            tensors.append((f"blk.{layer}.shexp_out.direction", v[None, :]))
        elif routed:
            raise SystemExit(f"layer {layer}: routed experts without the shared expert have no site")
    metadata = [
        ("general.architecture", arch),
        ("general.type", "adapter"),
        ("general.name", args.name or args.input.stem),
        ("adapter.type", "direction"),
        ("adapter.direction.mode", "project"),
        ("adapter.direction.scale", 1.0),
        ("adapter.direction.source", f"rank-1 LoRA {args.input.name}"),
    ]
    write_gguf(args.output, metadata, tensors)
    print(f"{args.output}: {len(tensors)} sets")


def inspect(args):
    src = Gguf(args.input)
    for key, value in src.kv.items():
        print(f"{key} = {value}")
    for name, t in src.tensors.items():
        v = src.f32(name).astype(np.float64)
        norms = np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=1)
        print(f"{name}: k={norms.size} norms {np.round(norms, 4).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("from-abl")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--arch", required=True, help="general.architecture of the model, e.g. deepseek41")
    p.add_argument("--name")
    p.set_defaults(run=from_abl)
    p = sub.add_parser("from-lora")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", type=Path, help="first shard of a base model build to check the LoRA's sign against")
    p.add_argument("--name")
    p.set_defaults(run=from_lora)
    p = sub.add_parser("inspect")
    p.add_argument("input", type=Path)
    p.set_defaults(run=inspect)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
