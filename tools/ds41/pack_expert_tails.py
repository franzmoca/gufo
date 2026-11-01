#!/usr/bin/env python3
"""Pack the tail of every routed expert of a DS41 GGUF into one file for a second drive.

With --stream-experts, a missing expert's gate, up and down projections are
read from the model's drive. When MODEL.gguf.expert-tails exists (the file
itself or a symlink to it on another NVMe drive), each projection's read is
split: the head comes from the GGUF and the tail from this file at the same
time, so the two drives' bandwidths add. The bytes are copied unchanged, so
results do not change.

The file keeps, for each projection, the bytes from a cut point (--head, the
smallest share of each projection read from the GGUF) to its end. An expert's
three tails are stored next to each other, each padded to 4 KiB, so direct
I/O reads one contiguous record. On btrfs the file is created without
copy-on-write (which also disables compression), as direct I/O needs.

Layout (little endian):
  header (8 KiB): magic "GUFOETL1", u32 version, u32 layers, u32 experts,
    u32 head share in millionths, then per shard
    (u32 count, then for each: u64 size), then per layer and projection:
    u64 expert bytes, u64 cut, u64 tail bytes padded, u64 offset of the
    projection's first expert in its shard, u32 shard index;
    per layer: u64 first record offset, u64 record bytes.
  records: layer L's expert e at first_record[L] + e * record_bytes[L].
"""
import argparse
import fcntl
import hashlib
import mmap
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.gguf import parse_gguf  # noqa: E402

BLOCK_BYTES = {16: 66, 17: 74, 18: 98, 19: 50, 20: 18 * 8, 21: 110, 22: 82, 23: 136, 29: 56,
               10: 84, 11: 110, 12: 144, 13: 176, 14: 210}
N_EXPERT = 384
N_LAYER = 40
ALIGN = 4096
HEADER = 8192
MAGIC = b"GUFOETL1"
VERSION = 1
FS_IOC_GETFLAGS = 0x80086601
FS_IOC_SETFLAGS = 0x40086602
FS_NOCOW_FL = 0x00800000
PROJ = ("gate", "up", "down")


def shards(first: Path) -> list[Path]:
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", str(first))
    if not match:
        raise SystemExit("expected the first GGUF shard, ending -00001-of-NNNNN.gguf")
    return [Path(f"{match[1]}-{i:05d}-of-{match[2]}.gguf") for i in range(1, int(match[2]) + 1)]


def expert_spans(paths: list[Path]):
    """{layer: [(shard, offset of expert 0, expert bytes)] for gate, up, down}."""
    spans = {}
    for index, path in enumerate(paths):
        gguf = parse_gguf(path)
        for tensor in gguf["tensors"]:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor["name"])
            if not match:
                continue
            experts, rows, cols = tensor["shape"]
            if experts != N_EXPERT or cols % 256 or tensor["type"] not in BLOCK_BYTES:
                raise SystemExit(f"unsupported expert tensor {tensor['name']} ({tensor['type_name']})")
            per_expert = rows * cols // 256 * BLOCK_BYTES[tensor["type"]]
            spans.setdefault(int(match[1]), [None] * 3)[PROJ.index(match[2])] = (
                index, gguf["data_offset"] + tensor["offset"], per_expert)
    if not spans or any(None in v for v in spans.values()):
        raise SystemExit("every MoE layer needs gate, up and down expert tensors")
    return spans


def pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def set_nocow(fd: int) -> bool:
    """Marks an empty file NOCOW on filesystems that support it (btrfs); False if not set."""
    try:
        flags = struct.unpack("l", fcntl.ioctl(fd, FS_IOC_GETFLAGS, struct.pack("l", 0)))[0]
        fcntl.ioctl(fd, FS_IOC_SETFLAGS, struct.pack("l", flags | FS_NOCOW_FL))
        flags = struct.unpack("l", fcntl.ioctl(fd, FS_IOC_GETFLAGS, struct.pack("l", 0)))[0]
        return bool(flags & FS_NOCOW_FL)
    except OSError:
        return False


def read_direct(fd: int, buf: mmap.mmap, offset: int, size: int) -> memoryview:
    start = offset // ALIGN * ALIGN
    length = pad(offset - start + size)
    view = memoryview(buf)[:length]
    done = 0
    while done < offset - start + size:
        got = os.preadv(fd, [view[done:]], start + done)
        if not got:
            raise OSError("short read")
        done += got
    return view[offset - start:offset - start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="first backbone GGUF shard")
    parser.add_argument("--out", type=Path, required=True,
                        help="output file on the second drive; link it as MODEL.gguf.expert-tails")
    parser.add_argument("--head", type=float, default=0.5,
                        help="smallest share of each projection read from the GGUF (default 0.5)")
    args = parser.parse_args()
    if not 0.0 < args.head < 1.0:
        parser.error("--head must be between 0 and 1")
    paths = shards(args.model)
    spans = expert_spans(paths)
    layers = sorted(spans)
    if layers != list(range(N_LAYER)):
        raise SystemExit(f"expected expert layers 0-{N_LAYER - 1}")

    # Per layer and projection: cut, padded tail; per layer: record bytes and offsets.
    table, first_record, record_bytes = {}, {}, {}
    offset = HEADER
    for layer in layers:
        record = 0
        for p, (shard, base, size) in enumerate(spans[layer]):
            cut = int(size * args.head)
            table[layer, p] = (size, cut, pad(size - cut), base, shard)
            record += pad(size - cut)
        first_record[layer], record_bytes[layer] = offset, record
        offset += record * N_EXPERT
    total = offset

    header = bytearray(HEADER)
    struct.pack_into("<8sIIII", header, 0, MAGIC, VERSION, N_LAYER, N_EXPERT, round(args.head * 1e6))
    pos = 24
    struct.pack_into("<I", header, pos, len(paths))
    pos += 4
    for path in paths:
        struct.pack_into("<Q", header, pos, path.stat().st_size)
        pos += 8
    for layer in range(N_LAYER):
        for p in range(3):
            size, cut, tail, base, shard = table.get((layer, p), (0, 0, 0, 0, 0))
            struct.pack_into("<QQQQI", header, pos, size, cut, tail, base, shard)
            pos += 36
    for layer in range(N_LAYER):
        struct.pack_into("<QQ", header, pos, first_record.get(layer, 0), record_bytes.get(layer, 0))
        pos += 16
    if pos > HEADER:
        raise SystemExit("header overflow")

    print(f"{len(layers)} layers, head share {args.head:.2f}: {total / 2**30:.2f} GiB to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    tmp.unlink(missing_ok=True)
    out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    if not set_nocow(out):
        print("note: could not mark the file NOCOW (fine on ext4/xfs; on btrfs direct reads would be buffered)")
    src = [os.open(p, os.O_RDONLY | os.O_DIRECT) for p in paths]
    buf = mmap.mmap(-1, 64 << 20)
    digest = hashlib.sha256()
    try:
        os.ftruncate(out, total)
        os.pwrite(out, bytes(header), 0)
        done = 0
        for layer in layers:
            for p in range(3):
                size, cut, tail, base, shard = table[layer, p]
                piece = sum(table[layer, q][2] for q in range(p))
                for e in range(N_EXPERT):
                    data = read_direct(src[shard], buf, base + e * size + cut, size - cut)
                    os.pwrite(out, data, first_record[layer] + e * record_bytes[layer] + piece)
                    if e in (0, N_EXPERT - 1):
                        digest.update(data)
                    data.release()
                    done += size - cut
            print(f"\r  layer {layer:2d}: {done / 2**30:6.1f} GiB", end="", flush=True)
        os.fsync(out)
    finally:
        buf.close()
        for fd in src:
            os.close(fd)
        os.close(out)
    os.replace(tmp, args.out)
    print(f"\ndone; sample digest {digest.hexdigest()[:16]}")
    link = Path(str(args.model) + ".expert-tails")
    print(f"to use it: ln -s {args.out.resolve()} {link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
