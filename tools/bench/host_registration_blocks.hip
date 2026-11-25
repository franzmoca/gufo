// Controlled neighbour-unmap reproduction for GPU-registered host buffers.
// Build: nix develop -c tools/bench/build.sh host_registration_blocks
// Reports host wall times; allocation/registration are outside timed scopes.
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
constexpr std::size_t block = 2ULL << 20;
constexpr std::size_t neighbour_bytes = 8ULL << 20;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
__global__ void Touch(unsigned* first, unsigned* last) {
  if (threadIdx.x == 0) {
    ++*first;
    ++*last;
  }
}
int main(int argc, char** argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 12;
  Check(hipSetDevice(0));
  hipDeviceProp_t properties{};
  Check(hipGetDeviceProperties(&properties, 0));
  std::fprintf(stderr, "device=%s arch=%s runtime=%d\n", properties.name,
               properties.gcnArchName, HIP_VERSION);
  hipStream_t stream{};
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  std::puts(
      "bytes,side,layout,repeat,control_ms,unmap_ms,post_unmap_ms,correct");
  for (const std::size_t bytes : {(128ULL << 20) + 4096, (1ULL << 30) + 4096}) {
    for (int side = 0; side < 2; ++side) {
      // ABBA layout order in each pair, fresh registration for every sample.
      for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int order = 0; order < 4; ++order) {
          const bool aligned = order == 1 || order == 2;
          const std::size_t mapped =
              aligned ? (bytes + block - 1) / block * block : bytes;
          const std::size_t reserve = mapped + 2 * neighbour_bytes + 2 * block;
          void* raw = mmap(nullptr, reserve, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
          if (raw == MAP_FAILED)
            throw std::runtime_error("reserve failed");
          const auto address = reinterpret_cast<std::uintptr_t>(raw);
          const auto base =
              (address + neighbour_bytes + block - 1) / block * block;
          auto* host =
              reinterpret_cast<unsigned char*>(base + (aligned ? 0 : 4096));
          void* neighbour = side == 0 ? host + mapped : host - neighbour_bytes;
          if (mmap(host, mapped, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != host ||
              mmap(neighbour, neighbour_bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != neighbour)
            throw std::runtime_error("mapping failed");
          if (madvise(host, mapped, MADV_NOHUGEPAGE) != 0)
            throw std::runtime_error("nohugepage failed");
          std::memset(host, 0, bytes);
          std::memset(neighbour, 1, neighbour_bytes);
          Check(hipHostRegister(host, bytes, hipHostRegisterDefault));
          void* device{};
          Check(hipHostGetDevicePointer(&device, host, 0));
          auto* first = static_cast<unsigned*>(device);
          auto* last = reinterpret_cast<unsigned*>(
              static_cast<unsigned char*>(device) + bytes - 4);
          auto run = [&] {
            const auto start = Clock::now();
            hipLaunchKernelGGL(Touch, dim3(1), dim3(32), 0, stream, first,
                               last);
            Check(hipGetLastError());
            Check(hipStreamSynchronize(stream));
            return Ms(start);
          };
          (void)run();
          const double control = run();
          const auto start = Clock::now();
          if (munmap(neighbour, neighbour_bytes) != 0)
            throw std::runtime_error("unmap failed");
          const double unmap = Ms(start);
          const double after = run();
          const bool correct =
              *reinterpret_cast<unsigned*>(host) == 3 &&
              *reinterpret_cast<unsigned*>(host + bytes - 4) == 3;
          std::printf("%zu,%s,%s,%d,%.6f,%.6f,%.6f,%d\n", bytes,
                      side == 0 ? "above" : "below",
                      aligned ? "isolated" : "shared-block", repeat, control,
                      unmap, after, correct);
          std::fflush(stdout);
          if (!correct)
            throw std::runtime_error("GPU write mismatch");
          Check(hipHostUnregister(host));
          if (munmap(raw, reserve) != 0)
            throw std::runtime_error("release failed");
        }
      }
    }
  }
  Check(hipStreamDestroy(stream));
}
