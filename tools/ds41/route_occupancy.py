#!/usr/bin/env python3
"""Report routed tile occupancy and export pp2048 counts for ds41_mmq_bench."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from prefetch_sim import read_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--counts', type=Path, help='Export 2048-row batches only')
    args = parser.parse_args()
    snapshot, prepares = read_trace(args.trace)
    writer = csv.writer(sys.stdout)
    writer.writerow(['prepare', 'layer', 'tokens', 'routes', 'experts',
                     'tiles64', 'live_groups16', 'slot_utilization', 'counts_line'])
    lines = []
    for index, prepare in enumerate(prepares):
        ids = prepare.ids.ravel()
        if np.any((ids < 0) | (ids >= snapshot.n_expert)):
            raise ValueError(f'Invalid expert ID in prepare {index}')
        counts = np.bincount(ids, minlength=snapshot.n_expert)
        tiles = int(((counts + 63) // 64).sum())
        groups = int(((counts + 15) // 16).sum())
        line = ''
        if prepare.n == 2048:
            line = len(lines)
            lines.append('COUNTS 2048 ' + ' '.join(map(str, counts)))
        writer.writerow([index, prepare.layer, prepare.n, len(ids),
                         int(np.count_nonzero(counts)), tiles, groups,
                         len(ids) / (64 * tiles) if tiles else 0, line])
    if args.counts:
        args.counts.write_text('\n'.join(lines) + ('\n' if lines else ''))


if __name__ == '__main__':
    main()
