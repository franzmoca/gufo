"""Compare per-expert reads: three projection spans (current GGUF) vs one
contiguous span of the same total size (expert-contiguous repack).

Usage: repack_io.py FIRST_SHARD.gguf EXPERTS_PER_LAYER ROUNDS split,contig[,seq]
"""
import mmap, os, random, re, sys, threading, time, concurrent.futures as cf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tools" / "ds41"))
sys.path.insert(0, str(ROOT / "tools"))
from stream_io_bench import spans  # noqa
A = 4096
first = sys.argv[1]; k = int(sys.argv[2]); rounds = int(sys.argv[3])
paths, _ = spans(first)
from gufo.gguf import parse_gguf
layers = {}
for fid, p in enumerate(paths):
    g = parse_gguf(p)
    for t in g["tensors"]:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", t["name"])
        if m:
            from stream_io_bench import BLOCK_BYTES
            size = t["shape"][1] * t["shape"][2] // 256 * BLOCK_BYTES[t["type"]]
            layers.setdefault(int(m[1]), []).append((fid, g["data_offset"] + t["offset"], size))
fds = [os.open(p, os.O_RDONLY | os.O_DIRECT) for p in paths]
bufs = threading.local()
def rd(job):
    fid, off, size = job
    a = off // A * A; n = (off + size - a + A - 1) // A * A
    b = getattr(bufs, "b", None)
    if b is None or len(b) < n:
        b = mmap.mmap(-1, max(n, 64 << 20)); bufs.b = b
    done = 0
    while done < n:
        got = os.preadv(fds[fid], [memoryview(b)[done:n]], a + done)
        if got <= 0: break
        done += got
    return n
def arm(kind, rng):
    total = 0; t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(4) as ex:
        for layer in sorted(layers):
            ts = layers[layer]; experts = rng.sample(range(384), k)
            if kind == "split":
                jobs = sorted((fid, base + e * size, size) for fid, base, size in ts for e in experts)
            elif kind == "seq":
                comb = sum(s for _, _, s in ts) * k; fid, base, size = min(ts, key=lambda x: x[1])
                start = base + rng.randrange(0, max(1, 384 * size - comb)) // A * A
                jobs = [(fid, start + i, min(32 << 20, comb - i)) for i in range(0, comb, 32 << 20)]
            else:
                comb = sum(s for _, _, s in ts); fid, base, size = min(ts, key=lambda x: x[1])
                region = 384 * size  # stay inside one projection block of this layer's shard
                jobs = sorted((fid, base + rng.randrange(0, max(1, region - comb)) // A * A, comb) for _ in experts)
            # split large reads at 32 MiB like the store
            total += sum(ex.map(rd, jobs))
    return total / (time.perf_counter() - t0) / 2**30
rng = random.Random(7)
for r in range(rounds):
    for kind in sys.argv[4].split(","):
        print(f"k={k} {kind}: {arm(kind, rng):.2f} GiB/s", flush=True)
