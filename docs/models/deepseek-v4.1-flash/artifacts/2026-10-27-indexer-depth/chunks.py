"""Per-chunk kernel-group times from a rocprofv3 db: splits the timeline at a
marker kernel (first dispatch of each chunk) and groups kernels by pattern."""
import collections
import re
import sqlite3
import sys

db, marker = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 8
c = sqlite3.connect(db)
rows = c.execute("""SELECT ks.display_name, kd.start, kd.end FROM rocpd_kernel_dispatch kd
 JOIN rocpd_info_kernel_symbol ks ON ks.id=kd.kernel_id ORDER BY kd.start""").fetchall()
short = lambda n: re.sub(r'\(.*', '', n.replace('(anonymous namespace)::', '')).replace('void ', '')
GROUPS = [
    ('idx-score', r'indexer_score'), ('topk', r'topk'), ('cand-mask', r'mask|block_max|compact_candidate'),
    ('attn', r'attention'), ('compress', r'compress'), ('dense16', r'dense_f16|ds41_dense'),
    ('routed16', r'routed'), ('mmq', r'mul_mat_q|mmq'), ('hc', r'hc_|mhc'), ('copy', r'copyBuffer|fillBuffer'),
]
def group(n):
    for g, p in GROUPS:
        if re.search(p, n, re.I): return g
    return 'other'
starts = [i for i, r in enumerate(rows) if marker in r[0]]
chunks = [rows[a:b] for a, b in zip(starts, starts[1:] + [len(rows)])]
names = [g for g, _ in GROUPS] + ['other']
print('chunk   span    busy ' + ' '.join('%9s' % g for g in names))
for i, ch in enumerate(chunks):
    if i % every and i != len(chunks) - 1: continue
    t = collections.Counter()
    for n, s, e in ch: t[group(short(n))] += e - s
    span = ch[-1][2] - ch[0][1]
    print('%4d %7.1f %7.1f ' % (i, span / 1e6, sum(t.values()) / 1e6) + ' '.join('%9.1f' % (t[g] / 1e6) for g in names))
if len(sys.argv) > 4:
    k = int(sys.argv[4]); t = collections.Counter(); cnt = collections.Counter()
    for n, s, e in chunks[k]: t[short(n)[:90]] += e - s; cnt[short(n)[:90]] += 1
    for n, v in t.most_common(40): print('%8.2f ms %5d  %s' % (v / 1e6, cnt[n], n))
