#!/usr/bin/env python3
"""Builds the DeepSeek V4.1 Flash vision sidecar from a ``deepseek41v`` mmproj.

The mmproj holds the vision tower and the image span embeddings, but not the
image routing bias: each MoE gate selects experts for image positions with
``bias_vl`` instead of its text ``bias`` (official inference/model.py, Gate).
That bias lives with the language model, which the GSQ-RCO quantizations do
not carry. This tool copies the mmproj and appends the 40 official vectors as
``blk.N.exp_probs_b_vl.bias`` (F32 [384]), the names of other DeepSeek V4.1
GGUFs. The bias is read from the official checkpoint: a local snapshot
directory, or range requests of only those tensors from Hugging Face.

    python3 -I tools/ds41/vision_sidecar.py --mmproj mmproj-BF16.gguf \\
        --out mmproj-ds41-BF16.gguf [--checkpoint DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gufo.gguf import Reader  # noqa: E402

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
LAYERS = 40
EXPERTS = 384
SOURCE_KEY = "deepseek41v.image_routing_bias.source"
F32 = 0


def checkpoint_reader(checkpoint: str | None):
    """Returns read(file, first, last) over the official safetensors shards."""
    if checkpoint:
        root = Path(checkpoint)

        def read(name: str, first: int, last: int) -> bytes:
            with (root / name).open("rb") as file:
                file.seek(first)
                data = file.read(last - first + 1)
            if len(data) != last - first + 1:
                raise ValueError(f"{name} is truncated")
            return data
        return read

    base = f"https://huggingface.co/{REPO}/resolve/main/"

    def read(name: str, first: int, last: int) -> bytes:
        request = urllib.request.Request(base + name, headers={"Range": f"bytes={first}-{last}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if len(data) != last - first + 1:
            raise ValueError(f"short range read of {name}")
        return data
    return read


def read_index(read, checkpoint: str | None) -> dict[str, str]:
    name = "model.safetensors.index.json"
    if checkpoint:
        return json.loads((Path(checkpoint) / name).read_text())["weight_map"]
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/main/{name}", timeout=60) as response:
        return json.loads(response.read())["weight_map"]


def routing_bias(checkpoint: str | None) -> np.ndarray:
    read = checkpoint_reader(checkpoint)
    index = read_index(read, checkpoint)
    headers: dict[str, tuple[int, dict]] = {}
    rows = []
    for layer in range(LAYERS):
        name = f"layers.{layer}.ffn.gate.bias_vl"
        shard = index.get(name)
        if shard is None:
            raise ValueError(f"checkpoint has no {name}")
        if shard not in headers:
            size = struct.unpack("<Q", read(shard, 0, 7))[0]
            headers[shard] = (size, json.loads(read(shard, 8, 8 + size - 1)))
        size, header = headers[shard]
        entry = header[name]
        if entry["dtype"] != "F32" or entry["shape"] != [EXPERTS]:
            raise ValueError(f"{name} is {entry['dtype']} {entry['shape']}, expected F32 [{EXPERTS}]")
        first, last = entry["data_offsets"]
        rows.append(np.frombuffer(read(shard, 8 + size + first, 8 + size + last - 1), dtype="<f4"))
    bias = np.stack(rows)
    if not np.isfinite(bias).all():
        raise ValueError("image routing bias is not finite")
    return bias


def gguf_string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def build(mmproj: Path, out: Path, bias: np.ndarray) -> None:
    data = mmproj.read_bytes()
    if data[:4] != b"GGUF":
        raise ValueError(f"{mmproj} is not a GGUF file")
    r = Reader(data)
    r.pos = 4
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kv_start = r.pos
    alignment = 32
    projector = None
    for _ in range(n_kv):
        key = r.string()
        tag = r.u32()
        if key == SOURCE_KEY:
            raise ValueError(f"{mmproj} already carries the image routing bias")
        if key in ("general.alignment", "clip.projector_type"):
            value = r.read_value(tag)
            if key == "general.alignment":
                alignment = value
            else:
                projector = value
        else:
            r.skip_value(tag)
    kv_end = r.pos
    if projector != "deepseek41v":
        raise ValueError(f"{mmproj} projector is {projector!r}, expected 'deepseek41v'")
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        infos.append((name, dims, r.u32(), r.u64()))
    data_start = (r.pos + alignment - 1) // alignment * alignment
    tensor_data = data[data_start:]

    def align(n: int) -> int:
        return (n + alignment - 1) // alignment * alignment

    # Existing tensor data stays in place; the vectors follow it.
    offset = align(len(tensor_data))
    added = []
    for layer in range(LAYERS):
        added.append((f"blk.{layer}.exp_probs_b_vl.bias", [EXPERTS], F32, offset))
        offset = align(offset + EXPERTS * 4)

    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", version, n_tensors + LAYERS, n_kv + 1)
    header += data[kv_start:kv_end]
    header += gguf_string(SOURCE_KEY) + struct.pack("<I", 8) + gguf_string(REPO)
    for name, dims, kind, at in infos + added:
        header += gguf_string(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", d) for d in dims)
        header += struct.pack("<IQ", kind, at)
    header += b"\0" * (align(len(header)) - len(header))

    body = bytearray(tensor_data)
    for layer, (_, _, _, at) in enumerate(added):
        body += b"\0" * (at - len(body))
        body += bias[layer].astype("<f4").tobytes()
    body += b"\0" * (align(len(body)) - len(body))
    tmp = out.with_name(out.name + ".partial")
    with tmp.open("wb") as file:
        file.write(header)
        file.write(body)
    tmp.replace(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmproj", required=True, type=Path, help="deepseek41v mmproj GGUF")
    parser.add_argument("--out", required=True, type=Path, help="sidecar to write")
    parser.add_argument("--checkpoint", help=f"local {REPO} snapshot (default: range requests)")
    args = parser.parse_args()
    if args.out.resolve() == args.mmproj.resolve():
        parser.error("--out must differ from --mmproj")
    bias = routing_bias(args.checkpoint)
    build(args.mmproj, args.out, bias)
    print(f"wrote {args.out}: {LAYERS} image routing bias vectors from {REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
