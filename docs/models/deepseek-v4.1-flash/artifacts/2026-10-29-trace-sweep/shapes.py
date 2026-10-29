import re
import sqlite3
import sys

db = sys.argv[1]
ntok = float(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
c = sqlite3.connect(db)
# decode window: after the last prefill routed WMMA launch
last = (
    c.execute("select max(end) from kernels where name like '%routed_f16%'").fetchone()[
        0
    ]
    or 0
)
rows = c.execute(
    """select name, grid_x/workgroup_x, grid_y/workgroup_y, grid_z/workgroup_z, count(*), sum(duration)/1e3, avg(duration)/1e3, vgpr_count, scratch_size
 from kernels where start > ? group by name, grid_x, grid_y, grid_z order by sum(duration) desc limit ?""",
    (last, top),
).fetchall()
tot = c.execute(
    "select sum(duration)/1e6, min(start), max(end) from kernels where start > ?",
    (last,),
).fetchone()
print(
    f"decode window: kernel {tot[0]:.1f} ms, span {(tot[2] - tot[1]) / 1e6:.1f} ms, per token {tot[0] / ntok:.2f} ms busy, {(tot[2] - tot[1]) / 1e6 / ntok:.2f} ms span"
)
for n, gx, gy, gz, cnt, tus, mus, v, sc in rows:
    n = re.sub(r"\(.*", "", re.sub(r"^void |\(anonymous namespace\)::", "", n))[:48]
    print(
        f"{n:48s} {gx:>6}x{gy}x{gz:<3} {cnt:6d} {cnt / ntok:6.1f}/tok {mus:8.1f}us {tus / ntok / 1e3:6.2f} ms/tok vgpr={v} scr={sc}"
    )
