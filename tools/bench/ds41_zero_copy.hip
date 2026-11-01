// Zero-copy expert slots on gfx1151: can the SSD DMA straight into memory
// the expert kernels read at full speed?
//
//   nix develop -c tools/bench/build.sh ds41_zero_copy
//   /tmp/ds41_zero_copy SHARD.gguf
//
// 1. Kernel read bandwidth over 6 random 8 MiB "experts" per iteration from a
//    2 GiB pool, per allocation type.
// 2. O_DIRECT reads of 8 MiB spans from SHARD into each host type, and a GPU
//    checksum against the CPU's after every read (stale cache lines).
// 3. Read-kernel slowdown while the current copy chain (pinned -> device ->
//    slot) or plain O_DIRECT DMA runs in the background.
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define CHECK(x)                                                                    \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

constexpr size_t kExpert = 8ull << 20;
constexpr size_t kPool = 2ull << 30;
constexpr int kSlots = kPool / kExpert;
constexpr int kPerIter = 6;

// Streams 6 experts with 16-byte loads, as the bandwidth-bound decode GEMVs do.
__global__ void read_experts(const uint4* const* experts, size_t words, uint32_t* out) {
  const int e = blockIdx.y;
  const uint4* p = experts[e];
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x + v.y * 3u + v.z * 5u + v.w * 7u;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;  // keeps the loads
}

__global__ void checksum(const uint32_t* p, size_t words, unsigned long long* out) {
  unsigned long long acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x)
    acc += (unsigned long long)p[i] * (i % 7 + 1);
  atomicAdd(out, acc);
}

enum Kind { kDevice, kManaged, kHostCoherent, kHostNonCoherent, kRegistered, kKinds };
const char* kNames[] = {"hipMalloc", "managed coarse (slots now)", "hipHostMalloc coherent",
                        "hipHostMalloc non-coherent", "malloc + hipHostRegister"};

struct Buffer {
  void* host = nullptr;  // CPU pointer (host kinds)
  void* dev = nullptr;   // GPU pointer
};

Buffer allocate(Kind kind, size_t bytes) {
  Buffer b;
  switch (kind) {
    case kDevice:
      CHECK(hipMalloc(&b.dev, bytes));
      break;
    case kManaged:
      CHECK(hipMallocManaged(&b.dev, bytes, hipMemAttachGlobal));
      (void)hipMemAdvise(b.dev, bytes, hipMemAdviseSetCoarseGrain, 0);
      (void)hipGetLastError();
      b.host = b.dev;
      break;
    case kHostCoherent:
      CHECK(hipHostMalloc(&b.host, bytes, hipHostMallocCoherent | hipHostMallocMapped));
      CHECK(hipHostGetDevicePointer(&b.dev, b.host, 0));
      break;
    case kHostNonCoherent:
      CHECK(hipHostMalloc(&b.host, bytes, hipHostMallocNonCoherent | hipHostMallocMapped));
      CHECK(hipHostGetDevicePointer(&b.dev, b.host, 0));
      break;
    case kRegistered:
      if (posix_memalign(&b.host, 2 << 20, bytes)) exit(1);
      memset(b.host, 1, bytes);
      CHECK(hipHostRegister(b.host, bytes, hipHostRegisterMapped));
      CHECK(hipHostGetDevicePointer(&b.dev, b.host, 0));
      break;
    default:
      break;
  }
  return b;
}

void release(Kind kind, Buffer b) {
  switch (kind) {
    case kDevice:
    case kManaged:
      (void)hipFree(b.dev);
      break;
    case kHostCoherent:
    case kHostNonCoherent:
      (void)hipHostFree(b.host);
      break;
    case kRegistered:
      (void)hipHostUnregister(b.host);
      free(b.host);
      break;
    default:
      break;
  }
}

// GB/s of read_experts over `iters` iterations of 6 random experts.
double read_bandwidth(const Buffer& pool, int iters, hipStream_t stream, uint32_t* out, const uint4** table,
                      std::mt19937& rng) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  std::vector<const uint4*> host_table(kPerIter * iters);
  for (auto& p : host_table)
    p = reinterpret_cast<const uint4*>(static_cast<char*>(pool.dev) + (rng() % kSlots) * kExpert);
  CHECK(hipMemcpy(table, host_table.data(), host_table.size() * sizeof(void*), hipMemcpyHostToDevice));
  const dim3 grid(256, kPerIter);
  read_experts<<<grid, 256, 0, stream>>>(table, kExpert / 16, out);  // warm-up
  CHECK(hipEventRecord(a, stream));
  for (int i = 0; i < iters; ++i)
    read_experts<<<grid, 256, 0, stream>>>(table + i * kPerIter, kExpert / 16, out);
  CHECK(hipEventRecord(b, stream));
  CHECK(hipEventSynchronize(b));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, a, b));
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return (double)iters * kPerIter * kExpert / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s SHARD.gguf\n", argv[0]);
    return 2;
  }
  const int fd = open(argv[1], O_RDONLY | O_DIRECT);
  if (fd < 0) {
    perror("open");
    return 1;
  }
  struct stat st;
  fstat(fd, &st);
  const size_t spans = (st.st_size - kExpert) / 4096;
  std::mt19937 rng(7);
  hipStream_t stream;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  uint32_t* out;
  CHECK(hipMalloc(&out, 64));
  const uint4** table;
  CHECK(hipMalloc(&table, 4096 * sizeof(void*)));
  unsigned long long* sum_dev;
  CHECK(hipMalloc(&sum_dev, 8));

  printf("1. kernel read bandwidth, 6 random 8 MiB experts per launch set, 2 GiB pool (GB/s, 3 runs)\n");
  for (int k = 0; k < kKinds; ++k) {
    Buffer pool = allocate(Kind(k), kPool);
    CHECK(hipMemset(pool.dev, 1, kPool));
    CHECK(hipDeviceSynchronize());
    printf("  %-28s", kNames[k]);
    for (int r = 0; r < 3; ++r) printf(" %7.1f", read_bandwidth(pool, 400, stream, out, table, rng));
    printf("\n");
    release(Kind(k), pool);
  }

  printf("2. O_DIRECT 8 MiB reads into host memory (GB/s, 6 parallel reads x 40) and GPU checksum checks\n");
  for (int k : {kManaged, kHostCoherent, kHostNonCoherent, kRegistered}) {
    Buffer buf = allocate(Kind(k), kPerIter * kExpert);
    double seconds = 0;
    int bad = 0;
    for (int it = 0; it < 40; ++it) {
      std::vector<off_t> offs(kPerIter);
      for (auto& o : offs) o = (off_t)(rng() % spans) * 4096;
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> threads;
      std::atomic<bool> failed{false};
      for (int j = 0; j < kPerIter; ++j)
        threads.emplace_back([&, j] {
          char* dst = static_cast<char*>(buf.host) + j * kExpert;
          size_t done = 0;
          while (done < kExpert) {
            const ssize_t got = pread(fd, dst + done, kExpert - done, offs[j] + done);
            if (got <= 0) {
              failed = true;
              return;
            }
            done += got;
          }
        });
      for (auto& t : threads) t.join();
      seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (failed) {
        printf("  %-28s O_DIRECT read failed: %s\n", kNames[k], strerror(errno));
        bad = -1;
        break;
      }
      unsigned long long cpu = 0;
      const uint32_t* words = static_cast<const uint32_t*>(buf.host);
      for (size_t i = 0; i < kPerIter * kExpert / 4; ++i) cpu += (unsigned long long)words[i] * (i % 7 + 1);
      CHECK(hipMemsetAsync(sum_dev, 0, 8, stream));
      checksum<<<512, 256, 0, stream>>>(static_cast<const uint32_t*>(buf.dev), kPerIter * kExpert / 4, sum_dev);
      unsigned long long gpu = 0;
      CHECK(hipMemcpyAsync(&gpu, sum_dev, 8, hipMemcpyDeviceToHost, stream));
      CHECK(hipStreamSynchronize(stream));
      bad += gpu != cpu;
    }
    if (bad >= 0)
      printf("  %-28s %6.2f GB/s, %d of 40 GPU checksums wrong\n", kNames[k],
             40.0 * kPerIter * kExpert / seconds / 1e9, bad);
    release(Kind(k), buf);
  }

  printf("3. read-kernel bandwidth on hipMalloc while background traffic runs (GB/s, 3 runs)\n");
  Buffer pool = allocate(kDevice, kPool);
  CHECK(hipMemset(pool.dev, 1, kPool));
  Buffer pinned = allocate(kHostNonCoherent, 8 * kExpert);
  Buffer stage = allocate(kDevice, 8 * kExpert);
  Buffer slots = allocate(kManaged, 8 * kExpert);
  hipStream_t copy_stream;
  CHECK(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
  for (int mode = 0; mode < 4; ++mode) {
    static const char* names[] = {"idle", "O_DIRECT DMA only (zero-copy path)",
                                  "O_DIRECT + H2D + slot copy (now)", "H2D + slot copy, no SSD"};
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> moved{0};
    std::thread bg([&] {
      std::mt19937 r2(11);
      while (!stop) {
        if (mode == 0) break;
        if (mode != 3) {
          size_t done = 0;
          const off_t off = (off_t)(r2() % spans) * 4096;
          while (done < kExpert) {
            const ssize_t got = pread(fd, static_cast<char*>(pinned.host) + done, kExpert - done, off + done);
            if (got <= 0) return;
            done += got;
          }
        }
        if (mode >= 2) {
          CHECK(hipMemcpyAsync(stage.dev, pinned.host, kExpert, hipMemcpyHostToDevice, copy_stream));
          CHECK(hipMemcpyAsync(slots.dev, stage.dev, kExpert, hipMemcpyDeviceToDevice, copy_stream));
          CHECK(hipStreamSynchronize(copy_stream));
        }
        moved += kExpert;
      }
    });
    const auto t0 = std::chrono::steady_clock::now();
    printf("  %-38s", names[mode]);
    for (int r = 0; r < 3; ++r) printf(" %7.1f", read_bandwidth(pool, 400, stream, out, table, rng));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop = true;
    bg.join();
    printf("   (background %.2f GB/s of experts)\n", moved / s / 1e9);
  }
  close(fd);
  return 0;
}
