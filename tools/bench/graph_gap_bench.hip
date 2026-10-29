// Dependent tiny-kernel chain: eager (null stream, explicit stream) vs graph replay.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ void step(float* a, const float* b, int n, int work) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = b[i];
  for (int k = 0; k < work; k++) v = v * 0.999f + 0.001f;
  a[i] = v;
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while (0)

int main() {
  const int launches = 1000;
  float *x, *y;
  const int n = 1 << 16;
  CK(hipMalloc(&x, n * 4)); CK(hipMalloc(&y, n * 4));
  hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  for (int work : {0, 64, 512}) {
    for (int blocks : {4, 64, 256}) {
      const int nn = blocks * 256;
      auto chain = [&](hipStream_t st) {
        for (int i = 0; i < launches; i++) step<<<blocks, 256, 0, st>>>(i & 1 ? x : y, i & 1 ? y : x, nn, work);
      };
      auto time = [&](auto fn, hipStream_t st) {
        fn(); hipStreamSynchronize(st); hipDeviceSynchronize();
        double best = 1e30;
        for (int r = 0; r < 20; r++) {
          auto t0 = std::chrono::steady_clock::now();
          fn(); hipStreamSynchronize(st); hipDeviceSynchronize();
          best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
      };
      double eager0 = time([&] { chain(0); }, 0);
      double eagerS = time([&] { chain(s); }, s);
      hipGraph_t g; hipGraphExec_t ge;
      CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
      chain(s);
      CK(hipStreamEndCapture(s, &g));
      CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      double graph = time([&] { hipGraphLaunch(ge, s); }, s);
      // single kernel time estimate
      double one = time([&] { step<<<blocks, 256, 0, s>>>(x, y, nn, work); }, s);
      printf("work %4d blocks %4d: eager-null %8.1f us  eager-stream %8.1f  graph %8.1f  (per launch %.2f / %.2f / %.2f us; single %.1f)\n",
             work, blocks, eager0, eagerS, graph, eager0 / launches, eagerS / launches, graph / launches, one);
      hipGraphExecDestroy(ge); hipGraphDestroy(g);
    }
  }
  return 0;
}
