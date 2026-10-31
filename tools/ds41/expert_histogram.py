#!/usr/bin/env python3
"""Build an optional expert-cache seed from DS41 oracle routes.npz files.

Write beside the first backbone shard as MODEL.gguf.expert-histogram; the
engine detects it when that model needs streaming. The input routes should
cover representative prompts for the same quant. Counts are performance
hints and do not change routing or numerical results.
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="First backbone GGUF shard")
    parser.add_argument("routes", type=Path, nargs="+", help="Oracle routes.npz files")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("the first backbone shard does not exist")
    counts = np.zeros((40, 384), dtype=np.uint64)
    for path in args.routes:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                routes = archive[name]
                if routes.ndim != 3 or routes.shape[0] != 40 or routes.shape[2] != 6:
                    raise ValueError(f"{path}:{name} is not [40, tokens, 6]")
                if not np.issubdtype(routes.dtype, np.integer) or np.any(routes < 0) or np.any(routes >= 384):
                    raise ValueError(f"{path}:{name} has invalid expert IDs")
                for layer in range(40):
                    counts[layer] += np.bincount(routes[layer].ravel().astype(np.int64), minlength=384).astype(np.uint64)
    output = Path(str(args.model) + ".expert-histogram")
    with output.open("x", encoding="ascii") as out:
        out.write("DS41_EXPERT_HISTOGRAM_V1\n")
        for layer in range(40):
            for expert in np.argsort(-counts[layer].astype(np.int64), kind="stable"):
                if counts[layer, expert]:
                    out.write(f"{layer} {expert} {counts[layer, expert]}\n")
    print(f"wrote {output}: {int(counts.sum())} routed uses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
