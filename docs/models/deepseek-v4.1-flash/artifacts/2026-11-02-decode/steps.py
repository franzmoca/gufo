"""Decode steps from a rocpd trace: split at the logits downloads (copies of k*517120 bytes)."""
import sqlite3, glob, sys, collections
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from names import nm
LOGIT = 129280 * 4
def load(d):
    db = glob.glob(f'{d}/*.db')[0]
    c = sqlite3.connect(db)
    ks = c.execute("""select ks.display_name,kd.start,kd.end,kd.grid_size_x,kd.grid_size_y,kd.stream_id
        from rocpd_kernel_dispatch kd join rocpd_info_kernel_symbol ks on ks.id=kd.kernel_id order by kd.start""").fetchall()
    names = {}
    K = []
    for n, s, e, gx, gy, st in ks:
        if n not in names: names[n] = nm(n)
        K.append((s, e, names[n], gx, gy, st))
    C = c.execute("select start,end,size,stream_id from rocpd_memory_copy order by start").fetchall()
    return K, C
def requests(K, C, gap=2.5e9):
    ev = sorted([(k[0], k[1]) for k in K] + [(x[0], x[1]) for x in C])
    out, s0, last = [], ev[0][0], ev[0][1]
    for s, e in ev[1:]:
        if s - last > gap: out.append((s0, last)); s0 = s
        last = max(last, e)
    out.append((s0, last))
    return out
