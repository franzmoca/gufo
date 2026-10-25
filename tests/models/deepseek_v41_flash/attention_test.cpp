// Batched (matrix-core) and split decode attention against a double
// precision reference of the plain softmax-with-sink over window and
// compressed rows; the indexer scores against the same kind of reference.
// The compressed rows are packed FP4, checked bit for bit against the
// fake-quantized floats through a host decoding of the documented format.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ds41_v41.h"

namespace {

constexpr uint32_t kHeads = 64;
constexpr uint32_t kDim = 512;
constexpr uint32_t kWindow = 128;
constexpr uint32_t kTopK = 512;
constexpr uint32_t kRope = 64;

template<typename T>
T* Device(const std::vector<T>& host) {
  void* p = nullptr;
  if (hipMalloc(&p, std::max<std::size_t>(host.size(), 1) * sizeof(T)) !=
      hipSuccess)
    return nullptr;
  (void)hipMemcpy(p, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice);
  return static_cast<T*>(p);
}

int g_format_errors = 0;

/* The stored FP4 row format decoded on the host, as documented with
 * ds41_fp4_rows. */
float HostFp4(const std::vector<uint8_t>& codes,
              const std::vector<uint8_t>& scales, std::size_t i, uint32_t block,
              bool e4m3) {
  static const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
  const uint32_t c = (codes[i / 2] >> (4 * (i & 1))) & 15u;
  const uint32_t b = scales[i / block];
  float scale;
  if (e4m3) {
    const uint32_t e = b >> 3, m = b & 7u;
    scale =
        e ? std::ldexp(1.0f + m / 8.0f, (int)e - 7) : std::ldexp((float)m, -9);
  } else {
    scale = std::ldexp(1.0f, (int)b - 127);
  }
  return (c & 8u ? -kE2m1[c & 7u] : kE2m1[c & 7u]) * scale;
}

/* Packs v as FP4 rows with ds41_v41_fp4_store, replaces v by the values of
 * ds41_v41_fp4_fake_quant (the reference), and counts in g_format_errors
 * the values whose host decoding of the packed bytes differs from them. */
ds41_fp4_rows Fp4(std::vector<float>& v, uint32_t block, bool e4m3,
                  uint8_t** codes_out, uint8_t** scales_out) {
  std::vector<uint8_t> codes(v.size() / 2), scales(v.size() / block);
  float* x = Device(v);
  uint8_t* dcodes = Device(codes);
  uint8_t* dscales = Device(scales);
  if (!x || !dcodes || !dscales ||
      ds41_v41_fp4_store(x, dcodes, dscales, (uint32_t)v.size(), block, e4m3) ||
      ds41_v41_fp4_fake_quant(x, (uint32_t)v.size(), block, e4m3) ||
      hipDeviceSynchronize() != hipSuccess) {
    g_format_errors++;
  }
  (void)hipMemcpy(v.data(), x, v.size() * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipMemcpy(codes.data(), dcodes, codes.size(), hipMemcpyDeviceToHost);
  (void)hipMemcpy(scales.data(), dscales, scales.size(), hipMemcpyDeviceToHost);
  (void)hipFree(x);
  for (std::size_t i = 0; i < v.size(); i++) {
    const float got = HostFp4(codes, scales, i, block, e4m3);
    if (std::memcmp(&got, &v[i], sizeof(float)) != 0)
      g_format_errors++;
  }
  *codes_out = dcodes;
  *scales_out = dscales;
  return {dcodes, dscales};
}

/* The FP8 rounding of ds41_v41_rope_fp8 on 32-value blocks. */
void HostFp8(float* v) {
  float amax = 0.0f;
  for (int i = 0; i < 32; i++)
    amax = std::max(amax, std::fabs(v[i]));
  int e = 0;
  const float m = std::frexp(std::max(amax, 1.0e-4f) / 448.0f, &e);
  const float scale = std::ldexp(1.0f, m == 0.5f ? e - 1 : e);
  for (int i = 0; i < 32; i++) {
    const float x = std::min(std::max(v[i] / scale, -448.0f), 448.0f);
    const float a = std::fabs(x);
    float step = 0.001953125f;
    if (a >= 0.015625f) {
      std::frexp(a, &e);
      step = std::ldexp(1.0f, e - 4);
    }
    v[i] =
        std::copysign(std::min(std::rint(a / step) * step, 448.0f), x) * scale;
  }
}

/* Rotates the tail kRope values of a row at position pos as
 * ds41_v41_rope_tail does (inverse: the opposite angle). */
void HostRope(float* row, const std::vector<float>& freq, uint32_t pos,
              bool inverse) {
  for (uint32_t i = 0; i < kRope / 2; i++) {
    const float angle = (float)pos * freq[i];
    const float c = std::cos(angle);
    const float s = inverse ? -std::sin(angle) : std::sin(angle);
    float* x = row + (kDim - kRope) + 2 * i;
    const float re = x[0], im = x[1];
    x[0] = re * c - im * s;
    x[1] = re * s + im * c;
  }
}

/* n tokens at pos0 with `comp_rows` compressed rows at the given ratio;
 * returns the worst relative RMS error over tokens. With `rope`, q is
 * rotated first and the output gets the inverse rotation and the FP8
 * rounding. */
double Case(uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t comp_rows,
            bool rope, std::mt19937& rng) {
  std::normal_distribution<float> normal;
  auto fill = [&](std::size_t count, float scale) {
    std::vector<float> v(count);
    for (auto& x : v)
      x = normal(rng) * scale;
    return v;
  };
  const auto q = fill((std::size_t)n * kHeads * kDim, 0.08f);
  const auto ring = fill((std::size_t)kWindow * kDim, 1.0f);
  const auto chunk = fill((std::size_t)n * kDim, 1.0f);
  auto comp = fill((std::size_t)std::max(comp_rows, 1u) * kDim, 1.0f);
  /* Rows of several magnitudes, down to subnormal E4M3 block scales. */
  for (std::size_t i = 0; i < comp.size(); i++)
    comp[i] *=
        (i / kDim) % 3 == 0 ? 1.0f : ((i / kDim) % 3 == 1 ? 1e-3f : 30.0f);
  uint8_t *dcodes = nullptr, *dscales = nullptr;
  const ds41_fp4_rows dcomp = Fp4(comp, 16, true, &dcodes, &dscales);
  const auto sinks = fill(kHeads, 1.0f);
  std::vector<uint32_t> idx((std::size_t)n * kTopK, 0);
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t len =
        ratio ? std::min(comp_rows, (pos0 + t + 1) / ratio) : 0;
    std::vector<uint32_t> rows(len);
    for (uint32_t j = 0; j < len; j++)
      rows[j] = j;
    std::shuffle(rows.begin(), rows.end(), rng);
    for (uint32_t j = 0; j < std::min(len, kTopK); j++)
      idx[(std::size_t)t * kTopK + j] = rows[j];
  }
  auto* dq = Device(q);
  auto* dring = Device(ring);
  auto* dchunk = Device(chunk);
  auto* dsinks = Device(sinks);
  auto* didx = Device(idx);
  std::vector<float> partial_host((std::size_t)n * kHeads *
                                  DS41_V41_ATTN_PARTIAL_PER_HEAD);
  auto* dpartial = Device(partial_host);
  std::vector<float> freq(kRope / 2);
  for (uint32_t i = 0; i < kRope / 2; i++)
    freq[i] = std::pow(160000.0f, -(float)(2 * i) / kRope);
  auto* dfreq = Device(freq);
  float *dout = nullptr, *dtable = nullptr;
  (void)hipMalloc(&dout, sizeof(float) * n * kHeads * kDim);
  (void)hipMalloc(&dtable, sizeof(float) * n * kRope);
  const float scale = 1.0f / std::sqrt((float)kDim);
  const int rc =
      ds41_v41_attention(dq, dring, dchunk, kWindow, dcomp, didx, kTopK, ratio,
                         dsinks, dout, dpartial, pos0, n, kHeads, kDim, scale,
                         rope ? dfreq : nullptr, kRope, dtable);
  std::vector<float> out((std::size_t)n * kHeads * kDim);
  (void)hipDeviceSynchronize();
  (void)hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
  for (void* p : {(void*)dq, (void*)dring, (void*)dchunk, (void*)dcodes,
                  (void*)dscales, (void*)dsinks, (void*)didx, (void*)dpartial,
                  (void*)dout, (void*)dfreq, (void*)dtable})
    (void)hipFree(p);
  if (rc != 0)
    return INFINITY;

  double worst = 0.0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t p = pos0 + t;
    const uint32_t lo = p + 1 > kWindow ? p + 1 - kWindow : 0;
    std::vector<const float*> keys;
    for (uint32_t pos = lo; pos <= p; pos++)
      keys.push_back(pos < pos0 ? &ring[(std::size_t)(pos % kWindow) * kDim]
                                : &chunk[(std::size_t)(pos - pos0) * kDim]);
    const uint32_t len =
        ratio ? std::min({comp_rows, (p + 1) / ratio, kTopK}) : 0;
    for (uint32_t j = 0; j < len; j++)
      keys.push_back(
          &comp[(std::size_t)idx[(std::size_t)t * kTopK + j] * kDim]);
    double err = 0.0, ref = 0.0;
    for (uint32_t h = 0; h < kHeads; h++) {
      std::vector<float> qrot(&q[((std::size_t)t * kHeads + h) * kDim],
                              &q[((std::size_t)t * kHeads + h + 1) * kDim]);
      if (rope)
        HostRope(qrot.data(), freq, p, false);
      const float* qh = qrot.data();
      std::vector<double> s(keys.size());
      double m = sinks[h];
      for (std::size_t j = 0; j < keys.size(); j++) {
        double dot = 0.0;
        for (uint32_t d = 0; d < kDim; d++)
          dot += (double)qh[d] * keys[j][d];
        s[j] = dot * scale;
        m = std::max(m, s[j]);
      }
      double den = std::exp(sinks[h] - m);
      for (auto& v : s)
        den += (v = std::exp(v - m));
      std::vector<float> o(kDim);
      for (uint32_t d = 0; d < kDim; d++) {
        double want = 0.0;
        for (std::size_t j = 0; j < keys.size(); j++)
          want += s[j] * keys[j][d];
        o[d] = (float)(want / den);
      }
      if (rope) {
        HostRope(o.data(), freq, p, true);
        for (uint32_t d = 0; d < kDim; d += 32)
          HostFp8(&o[d]);
      }
      for (uint32_t d = 0; d < kDim; d++) {
        const double want = o[d];
        const double got = out[((std::size_t)t * kHeads + h) * kDim + d];
        if (!std::isfinite(got))
          return INFINITY;
        err += (got - want) * (got - want);
        ref += want * want;
      }
    }
    worst = std::max(worst, std::sqrt(err / std::max(ref, 1e-30)));
  }
  return worst;
}

/* Indexer scores sum_h w[h] * scale * relu(q[h] . key[j]) for n tokens at
 * pos0 against double precision; returns the worst relative RMS error. */
double IndexerCase(uint32_t pos0, uint32_t n, uint32_t ratio,
                   std::mt19937& rng) {
  constexpr uint32_t heads = 32, dim = 128;
  const uint32_t len = (pos0 + n) / ratio;
  const uint64_t stride = len + 16;
  std::normal_distribution<float> normal;
  std::vector<float> q((std::size_t)n * heads * dim),
      keys((std::size_t)len * dim), w((std::size_t)n * heads);
  for (auto* v : {&q, &keys, &w})
    for (float& x : *v)
      x = normal(rng);
  const float scale = 0.0625f;
  float* d_q = Device(q);
  uint8_t *d_codes = nullptr, *d_scales = nullptr;
  const ds41_fp4_rows d_keys = Fp4(keys, 32, false, &d_codes, &d_scales);
  float* d_w = Device(w);
  float* d_score = nullptr;
  (void)hipMalloc(&d_score, (std::size_t)n * stride * sizeof(float));
  if (!d_q || !d_codes || !d_w || !d_score ||
      ds41_v41_indexer_score(d_q, d_keys, d_w, scale, d_score, stride, pos0, n,
                             ratio, heads, dim) ||
      hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "indexer launch failed\n");
    return INFINITY;
  }
  std::vector<float> score((std::size_t)n * stride);
  (void)hipMemcpy(score.data(), d_score, score.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
  (void)hipFree(d_codes);
  (void)hipFree(d_scales);
  for (float* p : {d_q, d_w, d_score})
    (void)hipFree(p);
  double worst = 0.0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t visible = (pos0 + t + 1) / ratio;
    double err = 0.0, ref = 0.0;
    for (uint32_t j = 0; j < visible; j++) {
      double acc = 0.0;
      for (uint32_t h = 0; h < heads; h++) {
        double dot = 0.0;
        for (uint32_t d = 0; d < dim; d++)
          dot += (double)q[((std::size_t)t * heads + h) * dim + d] *
                 keys[(std::size_t)j * dim + d];
        acc += std::max(dot, 0.0) * w[(std::size_t)t * heads + h] * scale;
      }
      const double diff = score[t * stride + j] - acc;
      err += diff * diff;
      ref += acc * acc;
    }
    worst = std::max(worst, std::sqrt(err / std::max(ref, 1e-30)));
  }
  return worst;
}

/* ds41_v41_q_kv_norm against ds41_v41_rms_norm and ds41_v41_rope_fp8 (the
 * same loops, so identical values; the tolerance only admits reassociation
 * under fast-math, well below an FP8 step); returns the largest absolute
 * difference. */
double QkvNormCase(uint32_t pos0, uint32_t n, std::mt19937& rng) {
  constexpr uint32_t q_dim = 1280, kv_dim = 512, rope = 64;
  std::normal_distribution<float> normal;
  std::vector<float> qa((std::size_t)n * q_dim), kv((std::size_t)n * kv_dim),
      qw(q_dim), kw(kv_dim), freq(rope / 2);
  for (auto* v : {&qa, &kv, &qw, &kw})
    for (float& x : *v)
      x = normal(rng);
  for (uint32_t i = 0; i < rope / 2; i++)
    freq[i] = std::pow(10000.0f, -2.0f * i / rope);
  float *d_qa = Device(qa), *d_kv = Device(kv), *d_qw = Device(qw),
        *d_kw = Device(kw), *d_freq = Device(freq);
  float *qr[2] = {}, *kvn[2] = {};
  for (int i = 0; i < 2; i++) {
    (void)hipMalloc(&qr[i], qa.size() * sizeof(float));
    (void)hipMalloc(&kvn[i], kv.size() * sizeof(float));
  }
  const float eps = 1e-6f;
  if (ds41_v41_rms_norm(d_qa, d_qw, qr[0], n, q_dim, eps) ||
      ds41_v41_rms_norm(d_kv, d_kw, kvn[0], n, kv_dim, eps) ||
      ds41_v41_rope_fp8(kvn[0], n, kv_dim, rope, d_freq, pos0, 1, 1, 0) ||
      ds41_v41_q_kv_norm(d_qa, d_qw, qr[1], q_dim, d_kv, d_kw, kvn[1], kv_dim,
                         rope, d_freq, pos0, n, eps) ||
      hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "q kv norm launch failed\n");
    return INFINITY;
  }
  double worst = 0.0;
  for (int which = 0; which < 2; which++) {
    const std::size_t count = which ? kv.size() : qa.size();
    std::vector<float> a(count), b(count);
    (void)hipMemcpy(a.data(), which ? kvn[0] : qr[0], count * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(b.data(), which ? kvn[1] : qr[1], count * sizeof(float),
                    hipMemcpyDeviceToHost);
    for (std::size_t i = 0; i < count; i++)
      worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
  }
  for (float* p :
       {d_qa, d_kv, d_qw, d_kw, d_freq, qr[0], qr[1], kvn[0], kvn[1]})
    (void)hipFree(p);
  return worst;
}

/* ds41_v41_matmul_bf16 (rows x k BF16 weights, n FP32 tokens) against
 * double precision; returns the worst error relative to the largest output
 * magnitude of a token. The batched form splits the activations into three
 * exact bfloat16 terms, so it stays at FP32 accumulation error. */
double Bf16MatmulCase(uint32_t rows, uint32_t k, uint32_t n,
                      std::mt19937& rng) {
  std::normal_distribution<float> normal;
  std::vector<uint16_t> w((std::size_t)rows * k);
  std::vector<float> wf(w.size()), x((std::size_t)n * k);
  for (std::size_t i = 0; i < w.size(); i++) {
    const float v = normal(rng) * 0.05f;
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    w[i] = (uint16_t)(u >> 16);
    u &= 0xffff0000u;
    std::memcpy(&wf[i], &u, sizeof(u));
  }
  for (std::size_t i = 0; i < x.size(); i++)
    x[i] = normal(rng) * (i % 7 == 0 ? 30.0f : 1.0f);
  uint16_t* dw = Device(w);
  float* dx = Device(x);
  float* dout = nullptr;
  (void)hipMalloc(&dout, (std::size_t)n * rows * sizeof(float));
  if (!dw || !dx || !dout || ds41_v41_matmul_bf16(dw, dx, dout, rows, k, n) ||
      hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "bf16 matmul launch failed\n");
    return INFINITY;
  }
  std::vector<float> out((std::size_t)n * rows);
  (void)hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(dout);
  double worst = 0.0;
  for (uint32_t t = 0; t < n; t++) {
    std::vector<double> want(rows);
    double scale = 1e-30;
    for (uint32_t r = 0; r < rows; r++) {
      double acc = 0.0;
      for (uint32_t i = 0; i < k; i++)
        acc += (double)wf[(std::size_t)r * k + i] * x[(std::size_t)t * k + i];
      want[r] = acc;
      scale = std::max(scale, std::fabs(acc));
    }
    for (uint32_t r = 0; r < rows; r++) {
      const double got = out[(std::size_t)t * rows + r];
      if (!std::isfinite(got))
        return INFINITY;
      worst = std::max(worst, std::fabs(got - want[r]) / scale);
    }
  }
  return worst;
}

}  // namespace

/* Times the attention of n tokens at pos0 (ratio 0, 1 or 2, top-k rows
 * drawn at random per token); returns the mean milliseconds per call. */
double Bench(uint32_t pos0, uint32_t n, uint32_t ratio, bool rope,
             std::mt19937& rng) {
  const uint32_t comp_rows = ratio ? (pos0 + n) / ratio : 0;
  std::vector<float> q((std::size_t)n * kHeads * kDim, 0.01f);
  std::vector<float> ring((std::size_t)kWindow * kDim, 0.5f);
  std::vector<float> chunk((std::size_t)n * kDim, 0.5f);
  std::vector<uint8_t> codes((std::size_t)std::max(comp_rows, 1u) * kDim / 2);
  std::vector<uint8_t> scales((std::size_t)std::max(comp_rows, 1u) * kDim / 16,
                              0x38);
  for (auto& c : codes)
    c = static_cast<uint8_t>(rng());
  std::vector<uint32_t> idx((std::size_t)n * kTopK, 0);
  std::vector<uint32_t> rows(comp_rows);
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t len = ratio ? (pos0 + t + 1) / ratio : 0;
    for (uint32_t j = 0; j < len; j++)
      rows[j] = j;
    std::shuffle(rows.begin(), rows.begin() + len, rng);
    for (uint32_t j = 0; j < std::min(len, kTopK); j++)
      idx[(std::size_t)t * kTopK + j] = rows[j];
  }
  std::vector<float> sinks(kHeads, 0.0f), freq(32);
  for (uint32_t i = 0; i < 32; i++)
    freq[i] = std::pow(10000.0f, -(float)i / 32.0f);
  auto* dq = Device(q);
  auto* dring = Device(ring);
  auto* dchunk = Device(chunk);
  auto* dcodes = Device(codes);
  auto* dscales = Device(scales);
  auto* didx = Device(idx);
  auto* dsinks = Device(sinks);
  auto* dfreq = Device(freq);
  float *dout = nullptr, *dtable = nullptr;
  (void)hipMalloc(&dout, sizeof(float) * n * kHeads * kDim);
  (void)hipMalloc(&dtable, sizeof(float) * n * 64);
  const float scale = 1.0f / std::sqrt((float)kDim);
  auto run = [&] {
    return ds41_v41_attention(dq, dring, dchunk, kWindow, {dcodes, dscales},
                              didx, kTopK, ratio, dsinks, dout, nullptr, pos0,
                              n, kHeads, kDim, scale, rope ? dfreq : nullptr,
                              64, dtable);
  };
  double ms = INFINITY;
  if (run() == 0 && hipDeviceSynchronize() == hipSuccess) {
    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    constexpr int kReps = 5;
    (void)hipEventRecord(start);
    for (int i = 0; i < kReps; i++)
      (void)run();
    (void)hipEventRecord(stop);
    (void)hipEventSynchronize(stop);
    float total = 0.0f;
    (void)hipEventElapsedTime(&total, start, stop);
    ms = total / kReps;
  }
  for (void* p :
       {(void*)dq, (void*)dring, (void*)dchunk, (void*)dcodes, (void*)dscales,
        (void*)didx, (void*)dsinks, (void*)dfreq, (void*)dout, (void*)dtable})
    (void)hipFree(p);
  return ms;
}

/* DSpark block attention: every query attends n_ring ring rows from
 * position `first` (slot pos % window) and all block rows, with sinks. */
double DsparkCase(uint32_t first, uint32_t n_ring, uint32_t n_block,
                  std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  auto fill = [&](std::size_t n, float scale) {
    std::vector<float> v(n);
    for (float& x : v)
      x = scale * dist(rng);
    return v;
  };
  const std::vector<float> q =
      fill((std::size_t)n_block * kHeads * kDim, 0.05f);
  const std::vector<float> ring = fill((std::size_t)kWindow * kDim, 1.0f);
  const std::vector<float> block = fill((std::size_t)n_block * kDim, 1.0f);
  const std::vector<float> sinks = fill(kHeads, 1.0f);
  std::vector<float> out(q.size());
  float* dq = Device(q);
  float* dring = Device(ring);
  float* dblock = Device(block);
  float* dsinks = Device(sinks);
  float* dout = Device(out);
  const float scale = 1.0f / std::sqrt((float)kDim);
  if (ds41_v41_dspark_attention(dq, dring, first, n_ring, kWindow, dblock,
                                n_block, dsinks, dout, kHeads, kDim, scale) ||
      hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return 1.0;
  for (float* p : {dq, dring, dblock, dsinks, dout})
    (void)hipFree(p);
  double err = 0.0, norm = 0.0;
  for (uint32_t t = 0; t < n_block; t++) {
    for (uint32_t h = 0; h < kHeads; h++) {
      const float* qh = &q[((std::size_t)t * kHeads + h) * kDim];
      std::vector<const float*> keys;
      for (uint32_t j = 0; j < n_ring; j++)
        keys.push_back(&ring[(std::size_t)((first + j) % kWindow) * kDim]);
      for (uint32_t j = 0; j < n_block; j++)
        keys.push_back(&block[(std::size_t)j * kDim]);
      std::vector<double> score(keys.size());
      double m = sinks[h];
      for (std::size_t j = 0; j < keys.size(); j++) {
        double dot = 0.0;
        for (uint32_t d = 0; d < kDim; d++)
          dot += (double)qh[d] * keys[j][d];
        score[j] = dot * scale;
        m = std::max(m, score[j]);
      }
      double sum = std::exp(sinks[h] - m);
      for (double& v : score)
        sum += v = std::exp(v - m);
      for (uint32_t d = 0; d < kDim; d++) {
        double ref = 0.0;
        for (std::size_t j = 0; j < keys.size(); j++)
          ref += score[j] * keys[j][d];
        ref /= sum;
        const double got = out[((std::size_t)t * kHeads + h) * kDim + d];
        err = std::max(err, std::abs(got - ref));
        norm = std::max(norm, std::abs(ref));
      }
    }
  }
  return err / norm;
}

/* hc mean into a strided row, and the biased argmax: first index of the
 * largest finite sum, non-finite values skipped, -1 when none is finite. */
bool DsparkHelpers(std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  constexpr uint32_t kHc = 4, kRowDim = 300, kRows = 3, kStride = 1000;
  std::vector<float> h(kRows * kHc * kRowDim);
  for (float& v : h)
    v = dist(rng);
  std::vector<float> out(kRows * kStride, 0.0f);
  float* dh = Device(h);
  float* dout = Device(out);
  bool ok = ds41_v41_hc_mean(dh, dout + 7, kStride, kHc, kRowDim, kRows) == 0 &&
            hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost) == hipSuccess;
  for (uint32_t t = 0; t < kRows && ok; t++)
    for (uint32_t i = 0; i < kRowDim; i++) {
      float sum = 0.0f;
      for (uint32_t c = 0; c < kHc; c++)
        sum += h[((std::size_t)t * kHc + c) * kRowDim + i];
      ok = ok && out[(std::size_t)t * kStride + 7 + i] == sum / kHc;
    }
  (void)hipFree(dh);
  (void)hipFree(dout);
  const uint32_t n = 129280;
  std::vector<float> x(n), bias(n);
  for (uint32_t i = 0; i < n; i++) {
    x[i] = dist(rng);
    bias[i] = dist(rng);
  }
  x[5000] = 40.0f;
  x[90000] =
      40.0f + bias[5000] - bias[90000];  // equal sum: the first index wins
  x[100] = INFINITY;
  x[200] = NAN;
  std::vector<int32_t> tokens(3, -2);
  float* dx = Device(x);
  float* dbias = Device(bias);
  int32_t* dtokens = Device(tokens);
  std::vector<float> none(16, NAN);
  float* dnone = Device(none);
  ok = ok && ds41_v41_argmax_bias(dx, dbias, n, dtokens, 1) == 0 &&
       ds41_v41_argmax_bias(dnone, nullptr, 16, dtokens, 2) == 0 &&
       hipMemcpy(tokens.data(), dtokens, tokens.size() * sizeof(int32_t),
                 hipMemcpyDeviceToHost) == hipSuccess;
  for (void* p : {(void*)dx, (void*)dbias, (void*)dtokens, (void*)dnone})
    (void)hipFree(p);
  std::printf(
      "dspark hc mean %s, argmax %d (expect 5000), none %d (expect -1)\n",
      ok ? "exact" : "MISMATCH", tokens[1], tokens[2]);
  return ok && tokens[1] == 5000 && tokens[2] == -1 && tokens[0] == -2;
}

/* F16 weights (the Markov tables) give the same sums as their exact F32
 * widening, and the F16 gather the same row. */
bool F16Matmul(std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  constexpr uint32_t kRows = 1000, kK = 256, kN = 3;
  std::vector<_Float16> w16(kRows * kK);
  std::vector<float> w32(w16.size()), x(kN * kK);
  for (std::size_t i = 0; i < w16.size(); i++) {
    w16[i] = (_Float16)(0.1f * dist(rng));
    w32[i] = (float)w16[i];
  }
  for (float& v : x)
    v = dist(rng);
  std::vector<float> a(kN * kRows), b(a.size()), row(kK), row_ref(kK);
  std::vector<int32_t> token = {417};
  _Float16* dw16 = Device(w16);
  float* dw32 = Device(w32);
  float* dx = Device(x);
  float* da = Device(a);
  float* db = Device(b);
  float* drow = Device(row);
  float* drow_ref = Device(row_ref);
  int32_t* dtoken = Device(token);
  bool ok =
      ds41_v41_matmul_f16((const uint16_t*)dw16, dx, da, kRows, kK, kN) == 0 &&
      ds41_v41_matmul_f32(dw32, dx, db, kRows, kK, kN) == 0 &&
      ds41_v41_gather_row(dw16, DS41_V41_ROW_F16, dtoken, 0, drow, kK) == 0 &&
      ds41_v41_gather_row(dw32, DS41_V41_ROW_F32, dtoken, 0, drow_ref, kK) ==
          0 &&
      hipMemcpy(a.data(), da, a.size() * sizeof(float),
                hipMemcpyDeviceToHost) == hipSuccess &&
      hipMemcpy(b.data(), db, b.size() * sizeof(float),
                hipMemcpyDeviceToHost) == hipSuccess &&
      hipMemcpy(row.data(), drow, kK * sizeof(float), hipMemcpyDeviceToHost) ==
          hipSuccess &&
      hipMemcpy(row_ref.data(), drow_ref, kK * sizeof(float),
                hipMemcpyDeviceToHost) == hipSuccess;
  for (void* p : {(void*)dw16, (void*)dw32, (void*)dx, (void*)da, (void*)db,
                  (void*)drow, (void*)drow_ref, (void*)dtoken})
    (void)hipFree(p);
  ok = ok && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0 &&
       std::memcmp(row.data(), row_ref.data(), kK * sizeof(float)) == 0;
  std::printf("f16 matmul and gather %s\n",
              ok ? "bit-identical to f32" : "MISMATCH");
  return ok;
}

int main(int argc, char** argv) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no HIP device\n");
    return 77;
  }
  /* --bench POS0 N RATIO [ROPE]: timing only. */
  if (argc >= 5 && !std::strcmp(argv[1], "--bench")) {
    std::mt19937 rng(7);
    const uint32_t pos0 = (uint32_t)std::atoi(argv[2]);
    const uint32_t n = (uint32_t)std::atoi(argv[3]);
    const uint32_t ratio = (uint32_t)std::atoi(argv[4]);
    const bool rope = argc >= 6 && std::atoi(argv[5]) != 0;
    std::printf("attention pos0=%u n=%u ratio=%u rope=%d %.3f ms\n", pos0, n,
                ratio, rope ? 1 : 0, Bench(pos0, n, ratio, rope, rng));
    return 0;
  }
  struct Shape {
    uint32_t pos0, n, ratio, comp_rows;
    bool rope;
    double tolerance;  // split decode is fp32; the batched form stages half
                       // operands; the FP8 rounding flips a few steps
  };
  const Shape shapes[] = {
      {12, 1, 0, 0, false, 1e-5},      {12, 1, 2, 6, false, 1e-5},
      {700, 1, 2, 350, false, 1e-5},   {2047, 1, 1, 2048, false, 1e-5},
      {3000, 3, 1, 3003, false, 1e-5}, {0, 37, 2, 18, false, 2e-3},
      {200, 64, 1, 264, false, 2e-3},  {2048, 40, 1, 2088, false, 2e-3},
      {300, 9, 0, 0, false, 2e-3},     {700, 1, 2, 350, true, 2e-2},
      {3000, 3, 1, 3003, true, 2e-2},  {0, 37, 2, 18, true, 2e-2},
      {2048, 40, 1, 2088, true, 2e-2},
  };
  std::mt19937 rng(41);
  bool ok = true;
  for (const Shape& s : shapes) {
    const double e = Case(s.pos0, s.n, s.ratio, s.comp_rows, s.rope, rng);
    std::printf("pos0=%u n=%u ratio=%u comp=%u rope=%d rel_err=%.2e\n", s.pos0,
                s.n, s.ratio, s.comp_rows, s.rope ? 1 : 0, e);
    ok = e < s.tolerance && ok;
  }
  /* Indexer scores: fp32 below five tokens, half operands on the matrix
   * cores above. */
  const Shape indexer[] = {
      {2047, 1, 1, 0, false, 1e-5},
      {101, 3, 2, 0, false, 1e-5},
      {3000, 4, 1, 0, false, 1e-5},
      {500, 20, 1, 0, false, 2e-3},
  };
  for (const Shape& s : indexer) {
    const double e = IndexerCase(s.pos0, s.n, s.ratio, rng);
    std::printf("indexer pos0=%u n=%u ratio=%u rel_err=%.2e\n", s.pos0, s.n,
                s.ratio, e);
    ok = e < s.tolerance && ok;
  }
  for (const uint32_t n : {1u, 37u}) {
    const double e = QkvNormCase(5000, n, rng);
    std::printf("q kv norm n=%u max_abs_diff=%.2e\n", n, e);
    ok = e < 1e-5 && ok;
  }
  /* Router, compressor and indexer shapes: FP32 kernel below 16 tokens, the
   * matrix cores above (tile edges at 37 and 200 tokens). */
  const uint32_t matmuls[][3] = {
      {384, 5120, 1}, {384, 5120, 37}, {512, 5120, 200}, {4096, 1280, 64}};
  for (const auto& m : matmuls) {
    const double e = Bf16MatmulCase(m[0], m[1], m[2], rng);
    std::printf("bf16 matmul rows=%u k=%u n=%u rel_err=%.2e\n", m[0], m[1],
                m[2], e);
    ok = e < 2e-6 && ok;
  }
  const uint32_t dspark[][3] = {{1743, 128, 5}, {0, 3, 5}, {60, 61, 2}};
  for (const auto& c : dspark) {
    const double e = DsparkCase(c[0], c[1], c[2], rng);
    std::printf("dspark attention first=%u ring=%u block=%u rel_err=%.2e\n",
                c[0], c[1], c[2], e);
    ok = e < 1e-5 && ok;
  }
  ok = DsparkHelpers(rng) && ok;
  ok = F16Matmul(rng) && ok;
  std::printf("fp4 row format mismatches %d\n", g_format_errors);
  ok = g_format_errors == 0 && ok;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
