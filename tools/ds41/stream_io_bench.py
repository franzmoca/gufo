#!/usr/bin/env python3
"""Read-only SSD benchmark using real routed-expert spans from a DS41 GGUF.

Each layer issues a small batch of exact gate/up/down projections, modeling
different numbers of cache misses. Run on an otherwise idle machine. Cold
arms ask the kernel to evict clean pages of these shards; no global system
settings or files are changed. O_DIRECT reads an aligned superset and checks
its exact payload against buffered reads before timing.
"""

import argparse
import concurrent.futures
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.gguf import parse_gguf  # noqa: E402

BLOCK_BYTES = {16: 66, 17: 74, 19: 50, 22: 82, 29: 56}
ALIGN = 4096


def spans(first):
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", str(first))
    if not match:
        raise ValueError("expected the first GGUF shard ending -00001-of-NNNNN.gguf")
    paths = [f"{match[1]}-{i:05d}-of-{match[2]}.gguf" for i in range(1, int(match[2]) + 1)]
    by_layer = {}
    for file_id, path in enumerate(paths):
        gguf = parse_gguf(path)
        for tensor in gguf["tensors"]:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight", tensor["name"])
            if not match:
                continue
            shape = tensor["shape"]
            if len(shape) != 3 or shape[0] != 384 or tensor["type"] not in BLOCK_BYTES:
                raise ValueError(f"unexpected expert tensor: {tensor}")
            size = shape[1] * shape[2] // 256 * BLOCK_BYTES[tensor["type"]]
            by_layer.setdefault(int(match[1]), []).append((file_id, gguf["data_offset"] + tensor["offset"], size))
    if len(by_layer) != 40 or any(len(t) != 3 for t in by_layer.values()):
        raise ValueError("expected three expert projections in each of 40 layers")
    rng = random.Random(41)
    batches = []
    for layer, tensors in sorted(by_layer.items()):
        experts = rng.sample(range(384), (1, 2, 6)[layer % 3])
        batch = sorted((fid, base + e * size, size) for fid, base, size in tensors for e in experts)
        batches.append(batch)
    return paths, batches


def read_into(fd, buf, offset, size, direct):
    start = offset // ALIGN * ALIGN if direct else offset
    skip = offset - start
    length = (skip + size + ALIGN - 1) // ALIGN * ALIGN if direct else size
    view = memoryview(buf)[:length]
    done = 0
    try:
        while done < skip + size:
            got = os.preadv(fd, [view[done:]], start + done)
            if not got:
                raise OSError("short read")
            done += got
            # An aligned superset can end past EOF. A short, unaligned final
            # read is valid if it includes the entire requested payload.
            if direct and got % ALIGN and done < skip + size:
                raise OSError("unaligned read before end of payload")
        return skip
    finally:
        view.release()


def drive_stats(path):
    if not path:
        return None
    values = list(map(int, Path(path).read_text().split()))
    return {"reads": values[0], "sectors": values[2], "read_ms": values[3]}


def run(paths, batches, mode, workers, stat_path, cold):
    fds = [os.open(p, os.O_RDONLY | (os.O_DIRECT if mode == "direct" else 0)) for p in paths]
    bufs = []
    local = threading.local()
    try:
        if cold:
            for fd in fds:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if mode == "random":
            for fd in fds:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        max_size = max(size for batch in batches for _, _, size in batch) + 2 * ALIGN
        def read(task):
            if not hasattr(local, "buf"):
                local.buf = mmap.mmap(-1, max_size)
                bufs.append(local.buf)
            fid, offset, size = task
            begin = time.perf_counter()
            read_into(fds[fid], local.buf, offset, size, mode == "direct")
            return time.perf_counter() - begin

        before = drive_stats(stat_path)
        latencies, batch_times = [], []
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for batch in batches:
                t = time.perf_counter()
                latencies.extend(pool.map(read, batch))
                batch_times.append(time.perf_counter() - t)
        seconds = time.perf_counter() - start
        after = drive_stats(stat_path)
        payload = sum(size for batch in batches for _, _, size in batch)
        latencies.sort()
        result = {"mode": mode, "workers": workers, "cold": cold, "calls": len(latencies),
                  "payload_bytes": payload, "seconds": seconds,
                  "payload_gib_per_s": payload / seconds / (1 << 30),
                  "read_ms_p50": statistics.median(latencies) * 1000,
                  "read_ms_p95": latencies[int(0.95 * (len(latencies) - 1))] * 1000,
                  "layer_ms_p50": statistics.median(batch_times) * 1000,
                  "layer_ms_p95": sorted(batch_times)[int(0.95 * (len(batch_times) - 1))] * 1000}
        if before is not None:
            result["device_read_bytes"] = (after["sectors"] - before["sectors"]) * 512
            result["device_reads"] = after["reads"] - before["reads"]
            result["device_read_ms"] = after["read_ms"] - before["read_ms"]
        return result
    finally:
        for buf in bufs:
            buf.close()
        for fd in fds:
            os.close(fd)


def check_bytes(paths, batches):
    checks = []
    tasks = [batch[0] for batch in batches[::7]] + [batches[-1][-1]]
    for fid, offset, size in tasks:
        with open(paths[fid], "rb") as f:
            expected = os.pread(f.fileno(), size, offset)
        fd = os.open(paths[fid], os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, size + 2 * ALIGN)
        try:
            skip = read_into(fd, buf, offset, size, True)
            actual = buf[skip:skip + size]
            if actual != expected:
                raise ValueError(f"direct read differs at {paths[fid]}:{offset}")
            checks.append(hashlib.sha256(actual).hexdigest())
        finally:
            buf.close()
            os.close(fd)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device-stat", help="optional /sys/block/<device>/stat")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    paths, batches = spans(args.model)
    result = {"model": str(args.model), "shards": paths, "checks_sha256": check_bytes(paths, batches),
              "batch_misses": [len(batch) // 3 for batch in batches], "arms": []}
    arms = [("buffered", 1), ("buffered", 4), ("random", 4), ("direct", 4), ("direct", 8), ("direct", 12)]
    for turn in range(args.rounds):
        for mode, workers in arms if turn % 2 == 0 else reversed(arms):
            sample = run(paths, batches, mode, workers, args.device_stat, cold=True)
            sample["round"] = turn + 1
            result["arms"].append(sample)
            print(json.dumps(sample), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
