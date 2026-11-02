#pragma clang fp contract(fast)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));exit(1);} }while(0)
constexpr uint32_t kBf16Tile = 64, kBf16K = 32, kBf16Pitch = kBf16K + 8;
__device__ __forceinline__ uint32_t bf16_rn_bits(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
template <uint32_t kRowTile>
__global__ __launch_bounds__(256) void matmul_bf16_wmma_kernel(const uint16_t* w, const float* x, float* out,
                                                               uint32_t rows, uint32_t k, uint32_t n) {
  constexpr int kFr = kRowTile / 32;  // 16-token fragments per wave
  __shared__ uint16_t w_sh[kRowTile * kBf16Pitch];
  __shared__ uint16_t x_sh[3][kBf16Tile * kBf16Pitch];
  const uint32_t tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const uint32_t r0 = blockIdx.x * kRowTile, t0 = blockIdx.y * kBf16Tile;
  /* Wave tile: weight rows wr .. wr + 15, tokens wt .. wt + 16 kFr - 1. */
  const uint32_t wr = (wave % (kRowTile / 16)) * 16, wt = (wave / (kRowTile / 16)) * 16 * kFr;
  using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, hip_bfloat16, rocwmma::row_major>;
  using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, hip_bfloat16, rocwmma::col_major>;
  using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
  float s[kFr][8] = {};
  for (uint32_t k0 = 0; k0 < k; k0 += kBf16K) {
    frag_c c[kFr];
#pragma unroll
    for (int f = 0; f < kFr; f++) rocwmma::fill_fragment(c[f], 0.0f);
    /* kRowTile rows x 32 values of W: uint4s (8 values) per thread. */
#pragma unroll
    for (uint32_t r = tid / 4; r < kRowTile; r += 64) {
      const uint32_t c = (tid % 4) * 8;
      *reinterpret_cast<uint4*>(&w_sh[r * kBf16Pitch + c]) =
          *reinterpret_cast<const uint4*>(w + (uint64_t)(r0 + r) * k + k0 + c);
    }
    for (uint32_t i = tid; i < kBf16Tile * kBf16K / 4; i += 256) {
      const uint32_t t = i / (kBf16K / 4), c = (i % (kBf16K / 4)) * 4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (t0 + t < n) v = *reinterpret_cast<const float4*>(x + (uint64_t)(t0 + t) * k + k0 + c);
      const float in[4] = {v.x, v.y, v.z, v.w};
      uint32_t hi[4], mid[4], lo[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        hi[e] = bf16_rn_bits(in[e]);
        const float r1 = in[e] - __uint_as_float(hi[e] << 16);
        mid[e] = bf16_rn_bits(r1);
        lo[e] = bf16_rn_bits(r1 - __uint_as_float(mid[e] << 16));
      }
      *reinterpret_cast<uint2*>(&x_sh[0][t * kBf16Pitch + c]) = make_uint2(hi[0] | hi[1] << 16, hi[2] | hi[3] << 16);
      *reinterpret_cast<uint2*>(&x_sh[1][t * kBf16Pitch + c]) =
          make_uint2(mid[0] | mid[1] << 16, mid[2] | mid[3] << 16);
      *reinterpret_cast<uint2*>(&x_sh[2][t * kBf16Pitch + c]) = make_uint2(lo[0] | lo[1] << 16, lo[2] | lo[3] << 16);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t kk = 0; kk < kBf16K; kk += 16) {
      frag_a a;
      rocwmma::load_matrix_sync(a, reinterpret_cast<const hip_bfloat16*>(w_sh + wr * kBf16Pitch + kk), kBf16Pitch);
#pragma unroll
      for (int part = 2; part >= 0; part--) {
#pragma unroll
        for (int f = 0; f < kFr; f++) {
          frag_b b;
          rocwmma::load_matrix_sync(
              b, reinterpret_cast<const hip_bfloat16*>(x_sh[part] + (wt + 16 * f) * kBf16Pitch + kk), kBf16Pitch);
          rocwmma::mma_sync(c[f], a, b, c[f]);
        }
      }
    }
#pragma unroll
    for (uint32_t e = 0; e < 8; e++)
#pragma unroll
      for (int f = 0; f < kFr; f++) s[f][e] += c[f].x[e];
    __syncthreads();
  }
  /* Accumulator element e of a lane is (row 2 * e + lane / 16, token lane % 16). */
#pragma unroll
  for (uint32_t e = 0; e < 8; e++) {
    const uint32_t row = r0 + wr + 2 * e + lane / 16;
#pragma unroll
    for (int f = 0; f < kFr; f++) {
      const uint32_t t = t0 + wt + 16 * f + lane % 16;
      if (t < n) out[(uint64_t)t * rows + row] = s[f][e];
    }
  }
}


constexpr uint32_t kBf16NarrowTokens = 16, kBf16NarrowWaves = 4, kBf16NarrowSteps = 4;
template <uint32_t kTokens, uint32_t kWaves, uint32_t kSteps>
__global__ __launch_bounds__(32 * kWaves) void matmul_bf16_wmma_narrow_kernel(const uint16_t* w, const float* x,
                                                                              float* out, uint32_t rows, uint32_t k,
                                                                              uint32_t n) {
  constexpr int kFr = kTokens / 16;
  constexpr uint32_t kThreadsN = 32 * kWaves, kGroupK = kBf16K * kSteps, kPitch = kGroupK + 8;
  constexpr uint32_t kXLoads = kTokens * kGroupK / 4 / kThreadsN;  // float4 per thread per group
  constexpr uint32_t kWLoads = 16 * kWaves * kGroupK / 8 / kThreadsN;        // uint4 per thread per group
  static_assert(kXLoads * kThreadsN == kTokens * kGroupK / 4 && kWLoads * kThreadsN * 8 == 16 * kWaves * kGroupK);
  __shared__ uint16_t w_sh[16 * kWaves * kPitch];
  __shared__ uint16_t x_sh[3][kTokens * kPitch];
  const uint32_t tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const uint32_t r0 = blockIdx.x * 16 * kWaves, t0 = blockIdx.y * kTokens;
  using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, hip_bfloat16, rocwmma::row_major>;
  using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, hip_bfloat16, rocwmma::col_major>;
  using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
  /* Group k0 covers K values k0 .. k0 + kGroupK - 1; the caller requires k % kGroupK == 0. */
  const auto load_w = [&](uint32_t k0, uint4 (&v)[kWLoads]) {
#pragma unroll
    for (uint32_t i = 0; i < kWLoads; i++) {
      const uint32_t q = tid + kThreadsN * i, r = q / (kGroupK / 8), c = (q % (kGroupK / 8)) * 8;
      v[i] = *reinterpret_cast<const uint4*>(w + (uint64_t)(r0 + r) * k + k0 + c);
    }
  };
  const auto load_x = [&](uint32_t k0, float4 (&v)[kXLoads]) {
#pragma unroll
    for (uint32_t i = 0; i < kXLoads; i++) {
      const uint32_t q = tid + kThreadsN * i, t = q / (kGroupK / 4), c = (q % (kGroupK / 4)) * 4;
      v[i] = t0 + t < n ? *reinterpret_cast<const float4*>(x + (uint64_t)(t0 + t) * k + k0 + c)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  };
  uint4 wv[kWLoads];
  float4 xv[kXLoads];
  load_w(0, wv);
  load_x(0, xv);
  float s[kFr][8] = {};
  for (uint32_t k0 = 0; k0 < k; k0 += kGroupK) {
#pragma unroll
    for (uint32_t i = 0; i < kWLoads; i++) {
      const uint32_t q = tid + kThreadsN * i;
      *reinterpret_cast<uint4*>(&w_sh[(q / (kGroupK / 8)) * kPitch + (q % (kGroupK / 8)) * 8]) = wv[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < kXLoads; i++) {
      const uint32_t q = tid + kThreadsN * i, t = q / (kGroupK / 4), c = (q % (kGroupK / 4)) * 4;
      const float in[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
      uint32_t hi[4], mid[4], lo[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        hi[e] = bf16_rn_bits(in[e]);
        const float r1 = in[e] - __uint_as_float(hi[e] << 16);
        mid[e] = bf16_rn_bits(r1);
        lo[e] = bf16_rn_bits(r1 - __uint_as_float(mid[e] << 16));
      }
      *reinterpret_cast<uint2*>(&x_sh[0][t * kPitch + c]) = make_uint2(hi[0] | hi[1] << 16, hi[2] | hi[3] << 16);
      *reinterpret_cast<uint2*>(&x_sh[1][t * kPitch + c]) = make_uint2(mid[0] | mid[1] << 16, mid[2] | mid[3] << 16);
      *reinterpret_cast<uint2*>(&x_sh[2][t * kPitch + c]) = make_uint2(lo[0] | lo[1] << 16, lo[2] | lo[3] << 16);
    }
    __syncthreads();
    if (k0 + kGroupK < k) {
      load_w(k0 + kGroupK, wv);
      load_x(k0 + kGroupK, xv);
    }
#pragma unroll
    for (uint32_t step = 0; step < kSteps; step++) {
      frag_c c[kFr];
#pragma unroll
      for (int f = 0; f < kFr; f++) rocwmma::fill_fragment(c[f], 0.0f);
#pragma unroll
      for (uint32_t kk = step * kBf16K; kk < (step + 1) * kBf16K; kk += 16) {
        frag_a a;
        rocwmma::load_matrix_sync(a, reinterpret_cast<const hip_bfloat16*>(w_sh + wave * 16 * kPitch + kk), kPitch);
#pragma unroll
        for (int part = 2; part >= 0; part--) {
#pragma unroll
          for (int f = 0; f < kFr; f++) {
            frag_b b;
            rocwmma::load_matrix_sync(b, reinterpret_cast<const hip_bfloat16*>(x_sh[part] + 16 * f * kPitch + kk),
                                      kPitch);
            rocwmma::mma_sync(c[f], a, b, c[f]);
          }
        }
      }
#pragma unroll
      for (uint32_t e = 0; e < 8; e++)
#pragma unroll
        for (int f = 0; f < kFr; f++) s[f][e] += c[f].x[e];
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t e = 0; e < 8; e++) {
    const uint32_t row = r0 + wave * 16 + 2 * e + lane / 16;
#pragma unroll
    for (int f = 0; f < kFr; f++) {
      const uint32_t t = t0 + 16 * f + lane % 16;
      if (t < n) out[(uint64_t)t * rows + row] = s[f][e];
    }
  }
}


template<unsigned Tile> float run(const uint16_t*w,const float*x,float*y,unsigned r,unsigned k,unsigned n){
 hipEvent_t a,b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
 for(int i=0;i<4;i++) { if constexpr(Tile==128) matmul_bf16_wmma_kernel<128><<<dim3(r/128,(n+63)/64),256>>>(w,x,y,r,k,n); else matmul_bf16_wmma_narrow_kernel<32,4,2><<<dim3(r/64,(n+31)/32),128>>>(w,x,y,r,k,n); }
 CHECK(hipEventRecord(a));
 for(int i=0;i<30;i++) { if constexpr(Tile==128) matmul_bf16_wmma_kernel<128><<<dim3(r/128,(n+63)/64),256>>>(w,x,y,r,k,n); else matmul_bf16_wmma_narrow_kernel<32,4,2><<<dim3(r/64,(n+31)/32),128>>>(w,x,y,r,k,n); }
 CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b)); float ms; CHECK(hipEventElapsedTime(&ms,a,b)); CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b)); return ms/30;
}
int main(){
 for(unsigned r: {384u,512u,4096u}) for(unsigned n:{65u,140u,2048u}){
 unsigned k=r==4096?1280:5120;
 std::mt19937 rng(42); std::uniform_real_distribution<float>d(-0.1f,0.1f);
 std::vector<uint16_t>w(r*k); std::vector<float>x(n*k),a(r*n),b(r*n);
 for(auto &v:w){float f=d(rng); unsigned u; memcpy(&u,&f,4); v=u>>16;} for(auto &v:x)v=d(rng);
 uint16_t*dw; float *dx,*da,*db; CHECK(hipMalloc(&dw,w.size()*2)); CHECK(hipMalloc(&dx,x.size()*4)); CHECK(hipMalloc(&da,a.size()*4)); CHECK(hipMalloc(&db,b.size()*4));
 CHECK(hipMemcpy(dw,w.data(),w.size()*2,hipMemcpyHostToDevice)); CHECK(hipMemcpy(dx,x.data(),x.size()*4,hipMemcpyHostToDevice));
 for(int repeat=0;repeat<3;repeat++){
 float ta=run<128>(dw,dx,da,r,k,n),tb=run<64>(dw,dx,db,r,k,n);
 CHECK(hipMemcpy(a.data(),da,a.size()*4,hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(),db,b.size()*4,hipMemcpyDeviceToHost));
 size_t bad=0; for(size_t i=0;i<a.size();i++)bad+=memcmp(&a[i],&b[i],4)!=0;
 printf("rows=%u k=%u n=%u repeat=%d tile128_ms=%.6f tile64_ms=%.6f mismatches=%zu\n",r,k,n,repeat,ta,tb,bad); if(bad)return 2;
 }
 CHECK(hipFree(dw));CHECK(hipFree(dx));CHECK(hipFree(da));CHECK(hipFree(db));
 }
}
