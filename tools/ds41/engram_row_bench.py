import os, random, time, threading, glob
path = glob.glob('/mnt/data/models/ds41/ista-gsq-rco-gguf/*Engram*')[0]
size = os.path.getsize(path)
nrows = size // 144
N = 100000
random.seed(7)
offs = [random.randrange(nrows) * 144 for _ in range(N)]

def drop():
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

def arm(label, nthread=32):
    drop()
    idx = [0]; lock = threading.Lock(); t0 = time.time(); done = [0]
    def work():
        fd = os.open(path, os.O_RDONLY)
        got = 0
        while True:
            with lock:
                i = idx[0]; idx[0] += 1
            if i >= N: break
            os.pread(fd, 144, offs[i])
            got += 1
        os.close(fd)
        with lock: done[0] += got
    th = [threading.Thread(target=work) for _ in range(nthread)]
    [t.start() for t in th]; [t.join() for t in th]
    dt = time.time() - t0
    print(f"{label}: {done[0]/dt/1000:.0f}K rows/s ({done[0]*4096/dt/2**30:.2f} GiB/s at 4K granule), mean {dt/done[0]*1e6:.0f} us/row")

print(f"{os.path.basename(path)}: {size/2**30:.1f} GiB, {nrows/1e6:.0f}M rows")
arm('cold buffered random 144B x32t')
arm('cold buffered random 144B x32t (repeat)')
