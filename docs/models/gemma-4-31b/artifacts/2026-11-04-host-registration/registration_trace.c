// Diagnostic interposer: record real HIP registrations without changing them.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
int hipHostRegister(void* pointer, size_t bytes, unsigned flags) {
  int (*original)(void*, size_t, unsigned) =
      dlsym(RTLD_NEXT, "hipHostRegister");
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  int result = original(pointer, bytes, flags);
  clock_gettime(CLOCK_MONOTONIC, &b);
  fprintf(stderr, "registration_trace bytes=%zu mod_2m=%zu result=%d ms=%.6f\n",
          bytes, (size_t)((uintptr_t)pointer % (2UL << 20)), result,
          (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1000000.0);
  return result;
}
