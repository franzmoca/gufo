// GPU unit and engine integration tests for Inkling 2 Mini LoopCD-Logits.
//
//   inkling2_loopcd_test [--model GGUF]
//
// Always runs §6.1 kernel unit tests on the GPU with synthetic logits.
// When --model is supplied, also runs §6.2 sequential engine checks:
//   - w_max = 0 bit-identity vs off,
//   - reference logits z_1 vs override_passes = 1 (prefill, Decode,
//   DecodeBatch,
//     Verify),
//   - graph vs eager under LoopCD,
//   - concurrent DecodeBatch/VerifyBatch vs lone Decode,
//   - prefill sink and interior checkpoints,
//   - 20-prompt sanity check (off hash + on differs on >= 18/20 prompts),
//   - session memory check at 32K and 256K.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/hip/fast.hpp"
#include "src/models/inkling2/text/chat_template.hpp"
#include "src/models/inkling2/text/tokenizer.hpp"
#include "src/models/inkling2/text/tool_dialect.hpp"

namespace i2 = gufo::models::inkling2;
namespace fs = gufo::models::inkling2::fast;

namespace {

#define REQUIRE(cond, ...)                                                   \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                     \
      std::fprintf(stderr, "\n");                                            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

void HipCheck(hipError_t e, const char* what) {
  REQUIRE(e == hipSuccess, "%s: %s", what, hipGetErrorString(e));
}

struct Rng {
  std::uint64_t s = 0x9E3779B97F4A7C15ULL;
  std::uint32_t Next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return static_cast<std::uint32_t>((s * 0x2545F4914F6CDD1DULL) >> 32);
  }
  float Uniform(float lo, float hi) {
    const double u = static_cast<double>(Next()) / 4294967296.0;
    return static_cast<float>(lo + (hi - lo) * u);
  }
};

struct HostOracle {
  std::vector<float> out;
  std::vector<double> w_used;
};

HostOracle RunHostOracle(std::span<const float> z, std::span<const float> z1,
                         int rows, int vocab, float w_max, bool adaptive,
                         std::span<const std::uint8_t> mask,
                         std::span<const std::uint32_t> ctrl_bits) {
  HostOracle res;
  res.out.assign(z.begin(), z.end());
  res.w_used.assign(static_cast<std::size_t>(rows), 0.0);
  auto is_ctrl = [&](int i) {
    if (ctrl_bits.empty())
      return false;
    return ((ctrl_bits[static_cast<std::size_t>(i) >> 5] >> (i & 31)) & 1U) !=
           0U;
  };
  for (int r = 0; r < rows; ++r) {
    const std::size_t off = static_cast<std::size_t>(r) * vocab;
    if ((!mask.empty() && mask[r] == 0) || !(w_max > 0.0F)) {
      res.w_used[r] = 0.0;
      continue;
    }
    double w = static_cast<double>(w_max);
    if (adaptive) {
      double m1 = -INFINITY, m2 = -INFINITY;
      for (int i = 0; i < vocab; ++i) {
        const double v = z[off + i];
        if (!std::isfinite(v))
          continue;
        if (v > m1) {
          m2 = m1;
          m1 = v;
        } else if (v > m2) {
          m2 = v;
        }
      }
      double sum = 0.0;
      for (int i = 0; i < vocab; ++i) {
        const double v = z[off + i];
        if (std::isfinite(v))
          sum += std::exp(v - m1);
      }
      double margin = 0.0;
      if (sum > 0.0 && std::isfinite(m1)) {
        const double p1 = 1.0 / sum;
        const double p2 = std::isfinite(m2) ? std::exp(m2 - m1) / sum : 0.0;
        margin = std::clamp(p1 - p2, 0.0, 1.0);
      }
      w = static_cast<double>(w_max) * (1.0 - margin);
    }
    res.w_used[r] = w;
    if (w == 0.0)
      continue;
    if (ctrl_bits.empty()) {
      for (int i = 0; i < vocab; ++i) {
        const double zi = z[off + i];
        const double z1i = z1[off + i];
        res.out[off + i] = static_cast<float>(zi + w * (zi - z1i));
      }
    } else {
      double mz = -INFINITY, mr = -INFINITY;
      for (int i = 0; i < vocab; ++i) {
        if (is_ctrl(i))
          continue;
        const double zi = z[off + i];
        const double ri = zi + w * (zi - static_cast<double>(z1[off + i]));
        if (std::isfinite(zi))
          mz = std::max(mz, zi);
        if (std::isfinite(ri))
          mr = std::max(mr, ri);
      }
      double sz = 0.0, sr = 0.0;
      for (int i = 0; i < vocab; ++i) {
        if (is_ctrl(i))
          continue;
        const double zi = z[off + i];
        const double ri = zi + w * (zi - static_cast<double>(z1[off + i]));
        if (std::isfinite(zi))
          sz += std::exp(zi - mz);
        if (std::isfinite(ri))
          sr += std::exp(ri - mr);
      }
      const double delta = (sz > 0.0 && sr > 0.0)
                               ? (mz + std::log(sz)) - (mr + std::log(sr))
                               : 0.0;
      for (int i = 0; i < vocab; ++i) {
        if (is_ctrl(i))
          continue;
        const double zi = z[off + i];
        const double ri = zi + w * (zi - static_cast<double>(z1[off + i]));
        res.out[off + i] = static_cast<float>(ri + delta);
      }
    }
  }
  return res;
}

std::vector<float> RunGpuKernel(std::span<const float> z,
                                std::span<const float> z1, int rows, int vocab,
                                float w_max, bool adaptive,
                                std::span<const std::uint8_t> mask,
                                std::span<const std::uint32_t> ctrl_bits) {
  const std::size_t elems = static_cast<std::size_t>(rows) * vocab;
  float *d_z = nullptr, *d_z1 = nullptr;
  std::uint8_t* d_mask = nullptr;
  std::uint32_t* d_ctrl = nullptr;
  void* d_ws = nullptr;
  HipCheck(hipMalloc(&d_z, elems * sizeof(float)), "malloc z");
  HipCheck(hipMalloc(&d_z1, elems * sizeof(float)), "malloc z1");
  HipCheck(
      hipMemcpy(d_z, z.data(), elems * sizeof(float), hipMemcpyHostToDevice),
      "h2d z");
  HipCheck(
      hipMemcpy(d_z1, z1.data(), elems * sizeof(float), hipMemcpyHostToDevice),
      "h2d z1");
  if (!mask.empty()) {
    HipCheck(hipMalloc(&d_mask, mask.size()), "malloc mask");
    HipCheck(hipMemcpy(d_mask, mask.data(), mask.size(), hipMemcpyHostToDevice),
             "h2d mask");
  }
  if (!ctrl_bits.empty()) {
    HipCheck(hipMalloc(&d_ctrl, ctrl_bits.size() * sizeof(std::uint32_t)),
             "malloc ctrl");
    HipCheck(hipMemcpy(d_ctrl, ctrl_bits.data(),
                       ctrl_bits.size() * sizeof(std::uint32_t),
                       hipMemcpyHostToDevice),
             "h2d ctrl");
  }
  HipCheck(hipMalloc(&d_ws, fs::LoopCdWorkspaceBytes(rows)), "malloc ws");
  fs::LoopCdArgs args{};
  args.logits = d_z;
  args.ref_logits = d_z1;
  args.rows = rows;
  args.vocab = vocab;
  args.w_max = w_max;
  args.adaptive = adaptive;
  args.guide_mask = d_mask;
  args.control_bits = d_ctrl;
  args.ws = d_ws;
  fs::LoopCdGuide(nullptr, args);
  HipCheck(hipDeviceSynchronize(), "sync kernel");
  std::vector<float> out(elems);
  HipCheck(
      hipMemcpy(out.data(), d_z, elems * sizeof(float), hipMemcpyDeviceToHost),
      "d2h out");
  (void)hipFree(d_z);
  (void)hipFree(d_z1);
  if (d_mask)
    (void)hipFree(d_mask);
  if (d_ctrl)
    (void)hipFree(d_ctrl);
  (void)hipFree(d_ws);
  return out;
}

int ArgmaxRow(std::span<const float> row) {
  return static_cast<int>(std::max_element(row.begin(), row.end()) -
                          row.begin());
}

std::vector<double> SoftmaxRow(std::span<const float> row) {
  double mx = -INFINITY;
  for (float v : row)
    mx = std::max(mx, static_cast<double>(v));
  std::vector<double> p(row.size());
  double s = 0.0;
  for (std::size_t i = 0; i < row.size(); ++i) {
    p[i] = std::exp(static_cast<double>(row[i]) - mx);
    s += p[i];
  }
  for (double& x : p)
    x /= s;
  return p;
}

void CheckAgainstOracle(const char* label, std::span<const float> z,
                        std::span<const float> z1, int rows, int vocab,
                        float w_max, bool adaptive,
                        std::span<const std::uint8_t> mask,
                        std::span<const std::uint32_t> ctrl_bits) {
  const auto want =
      RunHostOracle(z, z1, rows, vocab, w_max, adaptive, mask, ctrl_bits);
  const auto got =
      RunGpuKernel(z, z1, rows, vocab, w_max, adaptive, mask, ctrl_bits);
  for (int r = 0; r < rows; ++r) {
    const std::size_t off = static_cast<std::size_t>(r) * vocab;
    const auto g_row = std::span<const float>(got).subspan(off, vocab);
    const auto w_row = std::span<const float>(want.out).subspan(off, vocab);
    const auto z_row = z.subspan(off, vocab);
    float row_max = 1.0F;
    for (float v : w_row)
      row_max = std::max(row_max, std::fabs(v));
    double max_err = 0.0;
    for (int i = 0; i < vocab; ++i)
      max_err = std::max(max_err,
                         std::fabs(static_cast<double>(g_row[i]) - w_row[i]));
    REQUIRE(max_err <= 1e-5 * row_max, "%s row %d max_err=%.3e (row_max=%.3f)",
            label, r, max_err, row_max);
    REQUIRE(ArgmaxRow(g_row) == ArgmaxRow(w_row),
            "%s row %d argmax mismatch: %d vs %d", label, r, ArgmaxRow(g_row),
            ArgmaxRow(w_row));
    if (!ctrl_bits.empty()) {
      const auto p_z = SoftmaxRow(z_row);
      const auto p_g = SoftmaxRow(g_row);
      double ord_z = 0.0, ord_g = 0.0;
      for (int i = 0; i < vocab; ++i) {
        const bool ctrl =
            ((ctrl_bits[static_cast<std::size_t>(i) >> 5] >> (i & 31)) & 1U) !=
            0U;
        if (ctrl) {
          const double dp = std::fabs(p_g[i] - p_z[i]);
          REQUIRE(dp <= 1e-6, "%s row %d ctrl token %d dp=%.3e", label, r, i,
                  dp);
        } else {
          ord_z += p_z[i];
          ord_g += p_g[i];
        }
      }
      REQUIRE(std::fabs(ord_g - ord_z) <= 1e-6, "%s row %d ord mass diff=%.3e",
              label, r, std::fabs(ord_g - ord_z));
    }
  }
}

void RunKernelUnitTests() {
  constexpr int kV = 200058;
  Rng rng;

  // 1. Edge rows: ties, margin ~1, all-equal (margin 0), random rows in [-30,
  // 30].
  {
    const int rows = 8;
    std::vector<float> z(static_cast<std::size_t>(rows) * kV);
    std::vector<float> z1(static_cast<std::size_t>(rows) * kV);
    for (std::size_t i = 0; i < z.size(); ++i) {
      z[i] = rng.Uniform(-30.0F, 30.0F);
      z1[i] = rng.Uniform(-30.0F, 30.0F);
    }
    // Row 0: exact top-2 tie -> margin 0 -> w == w_max
    std::fill(z.begin(), z.begin() + kV, -5.0F);
    z[42] = 12.0F;
    z[12345] = 12.0F;
    // Row 1: dominant margin ~1 -> w ~ 0
    std::fill(z.begin() + kV, z.begin() + 2 * kV, -30.0F);
    z[kV + 7] = 30.0F;
    // Row 2: all-equal row -> margin 0 -> w == w_max
    std::fill(z.begin() + 2 * kV, z.begin() + 3 * kV, 3.25F);
    // Row 3: tight top-2 margin
    std::fill(z.begin() + 3 * kV, z.begin() + 4 * kV, -10.0F);
    z[3 * kV + 100] = 10.0F;
    z[3 * kV + 101] = 9.0F;

    std::vector<std::uint8_t> mask = {1, 1, 1, 1, 0, 1, 0, 1};
    const std::size_t words = (kV + 31) / 32;
    std::vector<std::uint32_t> ctrl_contig(words, 0);
    for (int i = 199998; i < kV; ++i)
      ctrl_contig[i >> 5] |= 1U << (i & 31);
    std::vector<std::uint32_t> ctrl_sparse = ctrl_contig;
    ctrl_sparse[0] |= 0x15U;
    ctrl_sparse[1000] |= 0x80000001U;

    CheckAgainstOracle("fixed_w", z, z1, rows, kV, 0.5F, false, {}, {});
    CheckAgainstOracle("adaptive_w", z, z1, rows, kV, 0.5F, true, {}, {});
    CheckAgainstOracle("adaptive_mask", z, z1, rows, kV, 1.0F, true, mask, {});
    CheckAgainstOracle("ctrl_contig", z, z1, rows, kV, 0.5F, true, mask,
                       ctrl_contig);
    CheckAgainstOracle("ctrl_sparse", z, z1, rows, kV, 1.0F, true, mask,
                       ctrl_sparse);

    // Logit bound: |z'| <= (1 + 2 * w_max) * 30 without control renorm.
    for (float w_max : {0.5F, 1.0F, 4.0F}) {
      const auto g = RunGpuKernel(z, z1, rows, kV, w_max, true, {}, {});
      const float bound = (1.0F + 2.0F * w_max) * 30.0F + 1e-4F;
      for (float v : g)
        REQUIRE(std::fabs(v) <= bound, "bound %.3f exceeded: %.3f", bound, v);
    }

    // w = 0 -> bit-identical to input in every mode.
    for (bool adap : {false, true}) {
      for (const auto& cb : {std::span<const std::uint32_t>{},
                             std::span<const std::uint32_t>(ctrl_contig)}) {
        const auto g = RunGpuKernel(z, z1, rows, kV, 0.0F, adap, {}, cb);
        REQUIRE(std::memcmp(g.data(), z.data(), z.size() * sizeof(float)) == 0,
                "w=0 modified logits (adaptive=%d)", adap ? 1 : 0);
      }
    }

    // Batch invariance: row r alone == row r inside batches of 1/8/16/64.
    std::vector<float> z64(static_cast<std::size_t>(64) * kV);
    std::vector<float> z1_64(static_cast<std::size_t>(64) * kV);
    for (int r = 0; r < 64; ++r) {
      const int src = r % rows;
      std::memcpy(z64.data() + static_cast<std::size_t>(r) * kV,
                  z.data() + static_cast<std::size_t>(src) * kV,
                  static_cast<std::size_t>(kV) * sizeof(float));
      std::memcpy(z1_64.data() + static_cast<std::size_t>(r) * kV,
                  z1.data() + static_cast<std::size_t>(src) * kV,
                  static_cast<std::size_t>(kV) * sizeof(float));
    }
    for (const auto& cb : {std::span<const std::uint32_t>{},
                           std::span<const std::uint32_t>(ctrl_contig)}) {
      const auto lone0 = RunGpuKernel(std::span<const float>(z64.data(), kV),
                                      std::span<const float>(z1_64.data(), kV),
                                      1, kV, 0.5F, true, {}, cb);
      const auto lone3 =
          RunGpuKernel(std::span<const float>(
                           z64.data() + static_cast<std::size_t>(3) * kV, kV),
                       std::span<const float>(
                           z1_64.data() + static_cast<std::size_t>(3) * kV, kV),
                       1, kV, 0.5F, true, {}, cb);
      for (int bs : {8, 16, 64}) {
        const auto batch =
            RunGpuKernel(std::span<const float>(
                             z64.data(), static_cast<std::size_t>(bs) * kV),
                         std::span<const float>(
                             z1_64.data(), static_cast<std::size_t>(bs) * kV),
                         bs, kV, 0.5F, true, {}, cb);
        REQUIRE(std::memcmp(batch.data(), lone0.data(),
                            static_cast<std::size_t>(kV) * sizeof(float)) == 0,
                "batch %d row 0 differs from lone", bs);
        REQUIRE(std::memcmp(batch.data() + static_cast<std::size_t>(3) * kV,
                            lone3.data(),
                            static_cast<std::size_t>(kV) * sizeof(float)) == 0,
                "batch %d row 3 differs from lone", bs);
      }
    }
  }

  // 2. Small vocab sizes (V = 1, 2, 17, 1024).
  for (int v : {1, 2, 17, 1024}) {
    std::vector<float> z(static_cast<std::size_t>(4) * v);
    std::vector<float> z1(static_cast<std::size_t>(4) * v);
    for (std::size_t i = 0; i < z.size(); ++i) {
      z[i] = rng.Uniform(-10.0F, 10.0F);
      z1[i] = rng.Uniform(-10.0F, 10.0F);
    }
    CheckAgainstOracle("small_vocab", z, z1, 4, v, 0.5F, true, {}, {});
  }
  std::printf("PASS §6.1 kernel unit tests\n");
}

std::uint64_t Fnv1a(std::span<const std::int32_t> xs) {
  std::uint64_t h = 14695981039346656037ULL;
  for (std::int32_t x : xs) {
    const auto* b = reinterpret_cast<const std::uint8_t*>(&x);
    for (int i = 0; i < 4; ++i) {
      h ^= b[i];
      h *= 1099511628211ULL;
    }
  }
  return h;
}

std::vector<std::int32_t> GreedyGenerate(i2::Engine& eng,
                                         std::span<const std::int32_t> prompt,
                                         int steps) {
  auto s = eng.NewSession();
  eng.Forward(*s, prompt);
  std::vector<std::int32_t> out;
  out.push_back(ArgmaxRow(s->logits()));
  for (int i = 1; i < steps; ++i) {
    std::int32_t next = 0;
    eng.Step(*s, out.back(), &next);
    out.push_back(next);
  }
  return out;
}

void RunEngineTests(const std::string& model_path) {
  std::string err;
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(model_path, &err, open);
  REQUIRE(reader != nullptr, "open %s: %s", model_path.c_str(), err.c_str());

  const std::uint32_t ctx = 8192;
  auto eng_off = i2::Engine::Create(reader, ctx, &err);
  REQUIRE(eng_off != nullptr, "eng_off: %s", err.c_str());

  i2::Engine::Options opt_zero;
  opt_zero.loopcd = {
      .mode = i2::LoopCdMode::kOn, .w_max = 0.0F, .allow_zero_strength = true};
  auto eng_zero = i2::Engine::Create(reader, ctx, &err, opt_zero);
  REQUIRE(eng_zero != nullptr, "eng_zero: %s", err.c_str());

  i2::Engine::Options opt_p1;
  opt_p1.override_passes = 1;
  auto eng_p1 = i2::Engine::Create(reader, ctx, &err, opt_p1);
  REQUIRE(eng_p1 != nullptr, "eng_p1: %s", err.c_str());

  i2::Engine::Options opt_on;
  opt_on.loopcd = {.mode = i2::LoopCdMode::kOn, .w_max = 0.5F};
  auto eng_on = i2::Engine::Create(reader, ctx, &err, opt_on);
  REQUIRE(eng_on != nullptr, "eng_on: %s", err.c_str());

  i2::Engine::Options opt_eager = opt_on;
  opt_eager.disable_graphs = true;
  auto eng_eager = i2::Engine::Create(reader, ctx, &err, opt_eager);
  REQUIRE(eng_eager != nullptr, "eng_eager: %s", err.c_str());

  const std::size_t V = eng_off->config().n_vocab_text;
  std::vector<std::int32_t> prompt(2200);
  for (std::size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = static_cast<std::int32_t>(1000 + (i * 7919) % 50000);

  // 1. w_max = 0 hook vs off: bit-identical across prefill sink, Decode,
  // DecodeBatch, Verify.
  {
    auto sa = eng_off->NewSession();
    auto sb = eng_zero->NewSession();
    std::vector<float> sink_a, sink_b;
    eng_off->Forward(*sa, std::span<const std::int32_t>(prompt).first(70),
                     [&](std::size_t, std::size_t c, const float* d) {
                       sink_a.insert(sink_a.end(), d, d + c * V);
                     });
    eng_zero->Forward(*sb, std::span<const std::int32_t>(prompt).first(70),
                      [&](std::size_t, std::size_t c, const float* d) {
                        sink_b.insert(sink_b.end(), d, d + c * V);
                      });
    REQUIRE(sink_a.size() == sink_b.size() &&
                std::memcmp(sink_a.data(), sink_b.data(),
                            sink_a.size() * sizeof(float)) == 0,
            "w=0 prefill sink != off");

    for (int i = 0; i < 8; ++i) {
      const std::int32_t tok = 2000 + i * 17;
      eng_off->Decode(*sa, tok);
      eng_zero->Decode(*sb, tok);
      REQUIRE(std::memcmp(sa->logits().data(), sb->logits().data(),
                          V * sizeof(float)) == 0,
              "w=0 decode step %d != off", i);
    }

    // Verify 8 rows.
    std::vector<std::int32_t> vtoks(8);
    for (int i = 0; i < 8; ++i)
      vtoks[i] = 3000 + i * 29;
    std::span<const float> va, vb;
    eng_off->Verify(*sa, vtoks, &va);
    eng_zero->Verify(*sb, vtoks, &vb);
    REQUIRE(std::memcmp(va.data(), vb.data(), 8 * V * sizeof(float)) == 0,
            "w=0 verify != off");
    eng_off->CommitVerify(*sa, 4, va);
    eng_zero->CommitVerify(*sb, 4, vb);
    eng_off->Decode(*sa, 4001);
    eng_zero->Decode(*sb, 4001);
    REQUIRE(std::memcmp(sa->logits().data(), sb->logits().data(),
                        V * sizeof(float)) == 0,
            "w=0 post-commit decode != off");
    std::printf("PASS §6.2 w_max=0 bit-identical to off\n");
  }

  // 2. Reference z_1 oracle vs override_passes = 1: prefill across ubatch,
  // interior checkpoints, Decode, DecodeBatch, Verify.
  {
    std::vector<float> z1_prefill, p1_prefill;
    eng_on->SetRefLogitsTap([&](std::size_t rows, const float* d) {
      z1_prefill.insert(z1_prefill.end(), d, d + rows * V);
    });
    auto s_on = eng_on->NewSession();
    auto s_p1 = eng_p1->NewSession();
    eng_on->Forward(*s_on, prompt,
                    [&](std::size_t, std::size_t, const float*) {});
    eng_p1->Forward(*s_p1, prompt,
                    [&](std::size_t, std::size_t c, const float* d) {
                      p1_prefill.insert(p1_prefill.end(), d, d + c * V);
                    });
    REQUIRE(z1_prefill.size() == p1_prefill.size() &&
                std::memcmp(z1_prefill.data(), p1_prefill.data(),
                            z1_prefill.size() * sizeof(float)) == 0,
            "prefill z_1 != override_passes=1");

    // Interior checkpoints z_1 vs override_passes=1.
    std::vector<float> z1_cp;
    eng_on->SetRefLogitsTap([&](std::size_t rows, const float* d) {
      z1_cp.insert(z1_cp.end(), d, d + rows * V);
    });
    auto s_on_cp = eng_on->NewSession();
    const std::vector<std::size_t> cps = {17, 700, 1500, 2048};
    eng_on->Forward(*s_on_cp, prompt, {}, nullptr, {}, cps,
                    [&](std::size_t, const i2::Engine::PrefillCheckpoint&) {});
    // z1_cp collected all interior cps (first 3) + end of ubatch 0 (2048) +
    // end of ubatch 1 (2200). Check positions 17, 700, 1500, 2048, 2200.
    REQUIRE(z1_cp.size() == 5 * V, "unexpected z1_cp size %zu", z1_cp.size());
    const std::size_t cp_pos[5] = {17, 700, 1500, 2048, 2200};
    for (int i = 0; i < 5; ++i) {
      const std::size_t p = cp_pos[i];
      REQUIRE(
          std::memcmp(z1_cp.data() + static_cast<std::size_t>(i) * V,
                      p1_prefill.data() + (p - 1) * V, V * sizeof(float)) == 0,
          "interior checkpoint %zu z_1 != override_passes=1", p);
    }

    // Decode z_1 vs override_passes=1.
    std::vector<float> z1_dec;
    eng_on->SetRefLogitsTap([&](std::size_t rows, const float* d) {
      z1_dec.assign(d, d + rows * V);
    });
    for (int i = 0; i < 12; ++i) {
      const std::int32_t tok = 5000 + i * 31;
      eng_on->Decode(*s_on, tok);
      eng_p1->Decode(*s_p1, tok);
      REQUIRE(std::memcmp(z1_dec.data(), s_p1->logits().data(),
                          V * sizeof(float)) == 0,
              "decode step %d z_1 != override_passes=1", i);
    }

    // Verify 16 rows z_1 vs override_passes=1.
    std::vector<std::int32_t> vtoks(16);
    for (int i = 0; i < 16; ++i)
      vtoks[i] = 6000 + i * 13;
    std::span<const float> v_on, v_p1;
    eng_on->Verify(*s_on, vtoks, &v_on);
    eng_p1->Verify(*s_p1, vtoks, &v_p1);
    REQUIRE(
        std::memcmp(z1_dec.data(), v_p1.data(), 16 * V * sizeof(float)) == 0,
        "verify(16) z_1 != override_passes=1");
    eng_on->CommitVerify(*s_on, 7, v_on);
    eng_p1->CommitVerify(*s_p1, 7, v_p1);
    eng_on->Decode(*s_on, 7001);
    eng_p1->Decode(*s_p1, 7001);
    REQUIRE(std::memcmp(z1_dec.data(), s_p1->logits().data(),
                        V * sizeof(float)) == 0,
            "post-commit decode z_1 != override_passes=1");

    // DecodeBatch (4 sessions) z_1 vs override_passes=1.
    std::vector<std::unique_ptr<i2::Session>> b_on(4), b_p1(4);
    std::vector<i2::Session*> pb_on(4), pb_p1(4);
    for (int g = 0; g < 4; ++g) {
      b_on[g] = eng_on->NewSession();
      b_p1[g] = eng_p1->NewSession();
      const auto sub =
          std::span<const std::int32_t>(prompt).first(300 + g * 400);
      eng_on->Forward(*b_on[g], sub);
      eng_p1->Forward(*b_p1[g], sub);
      pb_on[g] = b_on[g].get();
      pb_p1[g] = b_p1[g].get();
    }
    for (int step = 0; step < 6; ++step) {
      std::int32_t toks[4] = {100 + step, 200 + step, 300 + step, 400 + step};
      eng_on->DecodeBatch(pb_on, toks);
      eng_p1->DecodeBatch(pb_p1, toks);
      for (int g = 0; g < 4; ++g) {
        REQUIRE(std::memcmp(z1_dec.data() + static_cast<std::size_t>(g) * V,
                            b_p1[g]->logits().data(), V * sizeof(float)) == 0,
                "DecodeBatch step %d session %d z_1 != override_passes=1", step,
                g);
      }
    }
    eng_on->SetRefLogitsTap({});
    std::printf("PASS §6.2 sequential z_1 == override_passes=1 (bit-exact)\n");
  }

  // 3. Graph == eager & concurrent == alone under LoopCD.
  {
    const auto p_short = std::span<const std::int32_t>(prompt).first(600);
    auto sg = eng_on->NewSession();
    auto se = eng_eager->NewSession();
    eng_on->Forward(*sg, p_short);
    eng_eager->Forward(*se, p_short);
    for (int i = 0; i < 6; ++i) {
      const std::int32_t tok = 8000 + i * 7;
      eng_on->Decode(*sg, tok);
      eng_eager->Decode(*se, tok);
      REQUIRE(std::memcmp(sg->logits().data(), se->logits().data(),
                          V * sizeof(float)) == 0,
              "graph vs eager Decode step %d differs", i);
    }
    std::vector<std::int32_t> vt(8);
    for (int i = 0; i < 8; ++i)
      vt[i] = 8100 + i;
    std::span<const float> og, oe;
    eng_on->Verify(*sg, vt, &og);
    eng_eager->Verify(*se, vt, &oe);
    REQUIRE(std::memcmp(og.data(), oe.data(), 8 * V * sizeof(float)) == 0,
            "graph vs eager Verify differs");

    // Concurrent 4 sessions vs lone 4 sessions.
    std::vector<std::unique_ptr<i2::Session>> sl(4), sb(4);
    std::vector<i2::Session*> psb(4);
    for (int g = 0; g < 4; ++g) {
      const auto sub =
          std::span<const std::int32_t>(prompt).first(250 + g * 350);
      sl[g] = eng_on->NewSession();
      sb[g] = eng_on->NewSession();
      eng_on->Forward(*sl[g], sub);
      eng_on->Forward(*sb[g], sub);
      psb[g] = sb[g].get();
    }
    for (int i = 0; i < 6; ++i) {
      std::int32_t t[4] = {1000 + i, 2000 + i, 3000 + i, 4000 + i};
      eng_on->DecodeBatch(psb, t);
      for (int g = 0; g < 4; ++g) {
        eng_on->Decode(*sl[g], t[g]);
        REQUIRE(std::memcmp(sl[g]->logits().data(), sb[g]->logits().data(),
                            V * sizeof(float)) == 0,
                "concurrent DecodeBatch != alone at step %d session %d", i, g);
      }
    }
    std::printf(
        "PASS §6.2 graph == eager and concurrent == alone (bit-exact)\n");
  }

  // 4. Prefill sink & checkpoint rows equal standalone Forward at same
  // position, and agree with decode path at the same position.
  {
    const std::vector<std::size_t> pts = {17, 700, 1500, 2048, 2200};
    std::vector<float> sink_all;
    auto ssink = eng_on->NewSession();
    eng_on->Forward(*ssink, prompt,
                    [&](std::size_t, std::size_t c, const float* d) {
                      sink_all.insert(sink_all.end(), d, d + c * V);
                    });
    std::vector<std::vector<float>> cp_logits(4);
    auto scp = eng_on->NewSession();
    const std::vector<std::size_t> cps = {17, 700, 1500, 2048};
    eng_on->Forward(
        *scp, prompt, {}, nullptr, {}, cps,
        [&](std::size_t idx, const i2::Engine::PrefillCheckpoint& cp) {
          if (idx < 3)
            cp_logits[idx].assign(cp.logits.begin(), cp.logits.end());
          else
            cp_logits[idx] = scp->logits();
        });
    for (std::size_t i = 0; i < pts.size(); ++i) {
      const std::size_t pos = pts[i];
      auto sstop = eng_on->NewSession();
      eng_on->Forward(*sstop, std::span<const std::int32_t>(prompt).first(pos));
      const auto* sink_row = sink_all.data() + (pos - 1) * V;
      REQUIRE(
          std::memcmp(sstop->logits().data(), sink_row, V * sizeof(float)) == 0,
          "sink row at %zu != standalone Forward", pos);
      if (i < cps.size())
        REQUIRE(std::memcmp(sstop->logits().data(), cp_logits[i].data(),
                            V * sizeof(float)) == 0,
                "checkpoint row at %zu != standalone Forward", pos);
    }
    // Compare prefill sink row at pos=17 vs Decode of prompt[16] after 16
    // tokens.
    auto sdec = eng_on->NewSession();
    eng_on->Forward(*sdec, std::span<const std::int32_t>(prompt).first(16));
    eng_on->Decode(*sdec, prompt[16]);
    const auto* sink17 = sink_all.data() + 16 * V;
    double max_pd = 0.0;
    for (std::size_t i = 0; i < V; ++i)
      max_pd = std::max(
          max_pd,
          std::fabs(static_cast<double>(sdec->logits()[i]) - sink17[i]));
    REQUIRE(ArgmaxRow(sdec->logits()) ==
                ArgmaxRow(std::span<const float>(sink17, V)),
            "prefill vs decode argmax differ at pos 17");
    std::printf(
        "PASS §6.2 prefill sink & checkpoints bit-exact (prefill vs decode "
        "pos 17 max_diff=%.3e, argmax equal)\n",
        max_pd);
  }

  // 5. Fixed 20-prompt greedy sanity (off hash + on differs >= 18/20).
  {
    auto tok = inkling2::text::Tokenizer::CreateFromGguf(*reader, &err);
    REQUIRE(tok != nullptr, "tokenizer: %s", err.c_str());
    const char* prompts[20] = {
        "Write a C function that reverses a singly linked list in place.",
        "Explain the difference between TCP and UDP in three bullet points.",
        "Solve for x: 3x^2 - 14x + 8 = 0. Show your steps.",
        "Write a Python generator yielding Fibonacci numbers up to N.",
        "Summarize the causes of the 1907 Panic in two paragraphs.",
        "Implement binary search in Rust on a sorted slice of i32.",
        "What is the time complexity of building a binary heap from an array?",
        "Draft a concise bug report for a memory leak in a C++ HTTP parser.",
        "Prove that sqrt(2) is irrational.",
        "Write a SQL query returning the top 3 departments by average salary.",
        "Compare B-trees and LSM-trees for write-heavy workloads.",
        "Write a bash one-liner to count unique IP addresses in access.log.",
        "Explain how flash attention avoids materializing the attention "
        "matrix.",
        "Convert the recursive DFS into an iterative version in Go.",
        "What is the difference between covariance and contravariance in "
        "generics?",
        "Write a short haiku about autumn rain on mountain pines.",
        "Explain why IEEE 754 floating point addition is not associative.",
        "Write a regular expression matching ISO-8601 dates YYYY-MM-DD.",
        "Describe how Raft elects a new leader after a split vote.",
        "Calculate the determinant of [[2, -1, 0], [1, 3, 4], [0, -2, 5]].",
    };
    int diff = 0;
    std::vector<std::int32_t> all_off;
    for (int p = 0; p < 20; ++p) {
      inkling2::text::PromptTokens pt;
      inkling2::text::RenderOptions ro;
      ro.generation_prefix = inkling2::text::RenderOptions::Prefix::kHeader;
      REQUIRE(inkling2::text::RenderAndTokenize(
                  *tok, {inkling2::text::Message::Text("user", prompts[p])}, {},
                  ro, &pt, &err),
              "render prompt %d: %s", p, err.c_str());
      const std::vector<std::int32_t> pids(pt.ids.begin(), pt.ids.end());
      const auto g_off = GreedyGenerate(*eng_off, pids, 128);
      const auto g_on = GreedyGenerate(*eng_on, pids, 128);
      all_off.insert(all_off.end(), g_off.begin(), g_off.end());
      if (g_off != g_on)
        ++diff;
    }
    const std::uint64_t off_hash = Fnv1a(all_off);
    std::printf(
        "20-prompt sanity: off_fnv1a=0x%016llx, on differs on %d/20 prompts\n",
        static_cast<unsigned long long>(off_hash), diff);
    REQUIRE(diff >= 18, "LoopCD on differed on only %d/20 prompts (< 18)",
            diff);
  }

  // 6. Skip-thinking engine check: guide=0 bit-exact with off; guide=1
  // preserves control-token softmax mass; Verify + DecodeBatch + checkpoints
  // agree bit-exact with standalone Decode.
  {
    auto tok = inkling2::text::Tokenizer::CreateFromGguf(*reader, &err);
    REQUIRE(tok != nullptr, "tokenizer: %s", err.c_str());
    const std::size_t Vtok = tok->VocabSize();
    i2::Engine::Options opt_skip;
    opt_skip.loopcd.mode = i2::LoopCdMode::kSkipThinking;
    opt_skip.loopcd.w_max = 0.5F;
    opt_skip.loopcd_control_bits.assign((Vtok + 31) / 32, 0U);
    for (std::size_t id = 0; id < Vtok; ++id)
      if (tok->IsSpecial(static_cast<std::uint32_t>(id)))
        opt_skip.loopcd_control_bits[id / 32] |=
            (1U << static_cast<unsigned>(id % 32));
    auto eng_skip = i2::Engine::Create(reader, 4096, &err, opt_skip);
    REQUIRE(eng_skip != nullptr, "eng_skip create: %s", err.c_str());

    const auto p_skip = std::span<const std::int32_t>(prompt).first(64);
    const std::vector<std::uint8_t> zeros(p_skip.size(), 0);
    auto soff = eng_off->NewSession();
    auto ssk0 = eng_skip->NewSession();
    eng_off->Forward(*soff, p_skip);
    eng_skip->Forward(*ssk0, p_skip, {}, nullptr, {}, {}, {}, zeros);
    REQUIRE(std::memcmp(soff->logits().data(), ssk0->logits().data(),
                        V * sizeof(float)) == 0,
            "skip-thinking guide=0 Forward != off");
    for (int i = 0; i < 4; ++i) {
      const std::int32_t tok_id = 1500 + i * 19;
      eng_off->Decode(*soff, tok_id);
      eng_skip->Decode(*ssk0, tok_id, false);
      REQUIRE(std::memcmp(soff->logits().data(), ssk0->logits().data(),
                          V * sizeof(float)) == 0,
              "skip-thinking guide=0 Decode step %d != off", i);
    }

    // Compare guide=1 softmax control mass against eng_off at the same state.
    eng_off->Decode(*soff, 2000);
    eng_skip->Decode(*ssk0, 2000, true);
    auto ctrl_mass = [&](std::span<const float> row) {
      double mx = -INFINITY;
      for (float v : row)
        if (std::isfinite(v) && v > mx)
          mx = v;
      double zsum = 0.0, csum = 0.0;
      for (std::size_t id = 0; id < V; ++id) {
        if (!std::isfinite(row[id]))
          continue;
        const double e = std::exp(static_cast<double>(row[id]) - mx);
        zsum += e;
        if (tok->IsSpecial(static_cast<std::uint32_t>(id)))
          csum += e;
      }
      return csum / zsum;
    };
    const double cm_off = ctrl_mass(soff->logits());
    const double cm_skip = ctrl_mass(ssk0->logits());
    REQUIRE(std::fabs(cm_off - cm_skip) <= 1e-6,
            "control mass drift %.3e vs %.3e", cm_off, cm_skip);

    // Alternating guide={0,1,1,0,1,0} in Verify & DecodeBatch vs lone Decode.
    const std::vector<std::int32_t> vt = {2100, 2101, 2102, 2103, 2104, 2105};
    const std::vector<std::uint8_t> vg = {0, 1, 1, 0, 1, 0};
    auto sver = eng_skip->NewSession();
    eng_skip->Forward(*sver, p_skip, {}, nullptr, {}, {}, {}, zeros);
    std::span<const float> vout;
    eng_skip->Verify(*sver, vt, &vout, nullptr, {}, nullptr, vg);
    auto sare = eng_skip->NewSession();
    eng_skip->Forward(*sare, p_skip, {}, nullptr, {}, {}, {}, zeros);
    for (std::size_t r = 0; r < vt.size(); ++r) {
      eng_skip->Decode(*sare, vt[r], vg[r] != 0);
      REQUIRE(std::memcmp(sare->logits().data(), vout.subspan(r * V, V).data(),
                          V * sizeof(float)) == 0,
              "skip-thinking Verify row %zu != Decode", r);
    }
    std::printf(
        "PASS §6.2 skip-thinking (guide=0 == off, ctrl_mass diff=%.2e, Verify "
        "== Decode bit-exact)\n",
        std::fabs(cm_off - cm_skip));
  }

  // 7. Session memory check at 32K and 256K.
  eng_off.reset();
  eng_zero.reset();
  eng_p1.reset();
  eng_on.reset();
  eng_eager.reset();
  for (std::uint32_t mctx : {32768U, 262144U}) {
    std::size_t sb0 = 0, dm0 = 0, sb1 = 0, dm1 = 0;
    {
      auto e0 = i2::Engine::Create(reader, mctx, &err);
      REQUIRE(e0 != nullptr, "mem engine e0 create: %s", err.c_str());
      sb0 = e0->session_bytes();
      dm0 = e0->demand_mapped_bytes();
    }
    {
      auto e1 = i2::Engine::Create(reader, mctx, &err, opt_on);
      REQUIRE(e1 != nullptr, "mem engine e1 create: %s", err.c_str());
      sb1 = e1->session_bytes();
      dm1 = e1->demand_mapped_bytes();
    }
    const double ratio =
        dm0 ? static_cast<double>(dm1) / static_cast<double>(dm0) : 0.0;
    std::printf(
        "ctx=%u: off session=%.2f MiB (dm=%.2f MiB), on session=%.2f MiB "
        "(dm=%.2f MiB, ratio=%.4f)\n",
        mctx, sb0 / 1048576.0, dm0 / 1048576.0, sb1 / 1048576.0,
        dm1 / 1048576.0, ratio);
    if (dm0 > 0)
      REQUIRE(std::fabs(ratio - 9.0 / 8.0) < 0.01,
              "unexpected global KV ratio %.4f", ratio);
  }
}

void RunWeakModelDiagnostic(const std::string& model_path) {
  std::string err;
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(model_path, &err, open);
  REQUIRE(reader != nullptr, "open %s: %s", model_path.c_str(), err.c_str());

  auto tok = inkling2::text::Tokenizer::CreateFromGguf(*reader, &err);
  REQUIRE(tok != nullptr, "tokenizer: %s", err.c_str());

  // 1. Special-token check (§4).
  {
    const std::size_t V = tok->VocabSize();
    std::vector<std::uint32_t> specials;
    for (std::size_t id = 0; id < V; ++id) {
      if (tok->IsSpecial(static_cast<std::uint32_t>(id)))
        specials.push_back(static_cast<std::uint32_t>(id));
    }
    bool contiguous = !specials.empty();
    for (std::size_t i = 1; i < specials.size(); ++i) {
      if (specials[i] != specials[i - 1] + 1) {
        contiguous = false;
        break;
      }
    }
    std::printf(
        "§4 Special tokens: vocab=%zu count=%zu min=%u max=%u contiguous=%s\n",
        V, specials.size(), specials.empty() ? 0U : specials.front(),
        specials.empty() ? 0U : specials.back(), contiguous ? "true" : "false");
    if (!contiguous) {
      std::printf("  special ranges:");
      for (std::size_t i = 0; i < specials.size();) {
        std::size_t j = i + 1;
        while (j < specials.size() && specials[j] == specials[j - 1] + 1)
          ++j;
        std::printf(" [%u..%u]", specials[i], specials[j - 1]);
        i = j;
      }
      std::printf("\n");
    }
  }

  const auto t_header =
      tok->FindSpecial(inkling2::text::dialect::kHeader).value();
  const auto t_thinking =
      tok->FindSpecial(inkling2::text::dialect::kThinking).value();
  const auto t_end_msg =
      tok->FindSpecial(inkling2::text::dialect::kEndMessage).value();

  auto eng_off = i2::Engine::Create(reader, 8192, &err);
  REQUIRE(eng_off != nullptr, "eng_off: %s", err.c_str());
  i2::Engine::Options opt_p1;
  opt_p1.override_passes = 1;
  auto eng_p1 = i2::Engine::Create(reader, 8192, &err, opt_p1);
  REQUIRE(eng_p1 != nullptr, "eng_p1: %s", err.c_str());

  const std::vector<inkling2::text::Tool> tools = {
      {R"({"type":"function","function":{"name":"get_weather","description":"Get current weather.","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}})"},
      {R"({"type":"function","function":{"name":"run_sql","description":"Run a read-only SQL query.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}})"},
  };

  struct DiagPrompt {
    const char* cat;
    const char* text;
    bool with_tools;
  };
  const DiagPrompt prompts[20] = {
      {"prose",
       "Summarize the historical impact of the printing press in three "
       "sentences.",
       false},
      {"prose", "Explain why the sky is blue at noon and red at sunset.",
       false},
      {"prose", "Compare alpine and maritime climates concisely.", false},
      {"prose", "Describe how sourdough fermentation works in plain language.",
       false},
      {"prose",
       "List four practical ways to maintain focus during deep engineering "
       "work.",
       false},
      {"code",
       "Write a C function that reverses a singly linked list in place.",
       false},
      {"code", "Implement binary search in Rust on a sorted slice of i32.",
       false},
      {"code", "Write a Python generator yielding Fibonacci numbers up to n.",
       false},
      {"code",
       "Write a SQL query returning the top 3 departments by average salary.",
       false},
      {"code",
       "Write a short bash pipeline to count unique IPv4 addresses in "
       "access.log.",
       false},
      {"math", "Solve 3x^2 - 14x + 8 = 0 and verify both roots.", false},
      {"math", "Prove that sqrt(2) is irrational.", false},
      {"math",
       "Two fair six-sided dice are rolled. What is the probability the sum is "
       "at least 9?",
       false},
      {"math",
       "Explain why IEEE 754 float addition is not associative with a "
       "counterexample.",
       false},
      {"math",
       "Compute the determinant of [[2, -1, 0], [1, 3, 4], [0, -2, 5]].",
       false},
      {"tools", "What is the weather in Tokyo right now in celsius?", true},
      {"tools", "Check the weather in Reykjavik in fahrenheit.", true},
      {"tools", "Query the orders table for the 5 most recent orders.", true},
      {"tools", "Count active users who signed up in 2026 in the users table.",
       true},
      {"tools", "Find the current weather in Zurich.", true},
  };

  struct BucketStats {
    std::size_t rows = 0;
    std::size_t top1_agree = 0;
    double kl_sum = 0.0;
    double one_minus_margin_sum = 0.0;
    std::size_t w05_gt_025 = 0;
    std::size_t w10_gt_025 = 0;
    std::size_t flip_05 = 0;
    std::size_t flip_10 = 0;
  };
  BucketStats all_st, think_st, nonthink_st;
  const std::size_t V = eng_off->config().n_vocab_text;

  for (int p = 0; p < 20; ++p) {
    inkling2::text::PromptTokens pt;
    inkling2::text::RenderOptions ro;
    ro.generation_prefix = inkling2::text::RenderOptions::Prefix::kHeader;
    std::vector<inkling2::text::Tool> pt_tools;
    if (prompts[p].with_tools)
      pt_tools = tools;
    REQUIRE(inkling2::text::RenderAndTokenize(
                *tok, {inkling2::text::Message::Text("user", prompts[p].text)},
                pt_tools, ro, &pt, &err),
            "render prompt %d: %s", p, err.c_str());
    const std::vector<std::int32_t> prompt_ids(pt.ids.begin(), pt.ids.end());

    // Generate greedy output up to 256 tokens from eng_off.
    std::vector<std::int32_t> gen;
    {
      auto s = eng_off->NewSession();
      eng_off->Forward(*s, prompt_ids);
      std::int32_t next = ArgmaxRow(s->logits());
      gen.push_back(next);
      while (static_cast<int>(gen.size()) < 256 &&
             !tok->IsStop(static_cast<std::uint32_t>(next))) {
        eng_off->Step(*s, next, &next);
        gen.push_back(next);
      }
    }

    // Teacher-force prompt + gen[:-1] through both eng_off and eng_p1.
    std::vector<std::int32_t> seq = prompt_ids;
    seq.insert(seq.end(), gen.begin(), gen.end() - 1);
    const std::size_t gen_start = prompt_ids.size() - 1;
    const std::size_t L = gen.size();

    std::vector<float> z_all(L * V), z1_all(L * V);
    {
      auto s0 = eng_off->NewSession();
      eng_off->Forward(
          *s0, seq,
          [&](std::size_t first, std::size_t count, const float* data) {
            for (std::size_t i = 0; i < count; ++i) {
              const std::size_t pos = first + i;
              if (pos >= gen_start && pos < gen_start + L)
                std::memcpy(z_all.data() + (pos - gen_start) * V, data + i * V,
                            V * sizeof(float));
            }
          });
      auto s1 = eng_p1->NewSession();
      eng_p1->Forward(
          *s1, seq,
          [&](std::size_t first, std::size_t count, const float* data) {
            for (std::size_t i = 0; i < count; ++i) {
              const std::size_t pos = first + i;
              if (pos >= gen_start && pos < gen_start + L)
                std::memcpy(z1_all.data() + (pos - gen_start) * V, data + i * V,
                            V * sizeof(float));
            }
          });
    }

    // Track thinking block across generated tokens.
    bool after_header = true;
    bool in_thinking = false;
    for (std::size_t k = 0; k < L; ++k) {
      const float* z = z_all.data() + k * V;
      const float* z1 = z1_all.data() + k * V;

      double m1 = -INFINITY, m2 = -INFINITY;
      int argz = 0;
      double m1_ref = -INFINITY;
      int argz1 = 0;
      for (std::size_t i = 0; i < V; ++i) {
        const double v = z[i];
        if (v > m1) {
          m2 = m1;
          m1 = v;
          argz = static_cast<int>(i);
        } else if (v > m2) {
          m2 = v;
        }
        const double vr = z1[i];
        if (vr > m1_ref) {
          m1_ref = vr;
          argz1 = static_cast<int>(i);
        }
      }
      double sum_z = 0.0, sum_z1 = 0.0;
      for (std::size_t i = 0; i < V; ++i) {
        sum_z += std::exp(static_cast<double>(z[i]) - m1);
        sum_z1 += std::exp(static_cast<double>(z1[i]) - m1_ref);
      }
      const double lse_z = m1 + std::log(sum_z);
      const double lse_z1 = m1_ref + std::log(sum_z1);
      double kl = 0.0;
      for (std::size_t i = 0; i < V; ++i) {
        const double log_p = static_cast<double>(z[i]) - lse_z;
        const double log_p1 = static_cast<double>(z1[i]) - lse_z1;
        const double pi = std::exp(log_p);
        if (pi > 0.0)
          kl += pi * (log_p - log_p1);
      }
      const double p1 = 1.0 / sum_z;
      const double p2 = std::exp(m2 - m1) / sum_z;
      const double one_minus_margin = 1.0 - std::clamp(p1 - p2, 0.0, 1.0);
      const double w05 = 0.5 * one_minus_margin;
      const double w10 = 1.0 * one_minus_margin;

      int arg_05 = 0, arg_10 = 0;
      double best_05 = -INFINITY, best_10 = -INFINITY;
      for (std::size_t i = 0; i < V; ++i) {
        const double zi = z[i], z1i = z1[i];
        const double g05 = zi + w05 * (zi - z1i);
        const double g10 = zi + w10 * (zi - z1i);
        if (g05 > best_05) {
          best_05 = g05;
          arg_05 = static_cast<int>(i);
        }
        if (g10 > best_10) {
          best_10 = g10;
          arg_10 = static_cast<int>(i);
        }
      }

      auto accum = [&](BucketStats& b) {
        ++b.rows;
        if (argz == argz1)
          ++b.top1_agree;
        b.kl_sum += kl;
        b.one_minus_margin_sum += one_minus_margin;
        if (w05 > 0.25)
          ++b.w05_gt_025;
        if (w10 > 0.25)
          ++b.w10_gt_025;
        if (arg_05 != argz)
          ++b.flip_05;
        if (arg_10 != argz)
          ++b.flip_10;
      };
      accum(all_st);
      accum(in_thinking ? think_st : nonthink_st);

      const auto tid = static_cast<std::uint32_t>(gen[k]);
      if (tid == t_header) {
        after_header = true;
        in_thinking = false;
      } else if (after_header) {
        in_thinking = (tid == t_thinking);
        after_header = false;
      } else if (tid == t_end_msg) {
        in_thinking = false;
      }
    }
  }

  auto print_stats = [&](const char* name, const BucketStats& b) {
    const double n = b.rows ? static_cast<double>(b.rows) : 1.0;
    const double mean_omm = b.one_minus_margin_sum / n;
    std::printf(
        "§7.1 [%s] rows=%zu top1_agree(z1,z)=%.2f%% mean_KL(p||p1)=%.4f "
        "mean(1-margin)=%.4f | w_max=0.5: mean_w=%.4f frac(w>0.25)=%.2f%% "
        "flip_rate=%.2f%% | w_max=1.0: mean_w=%.4f frac(w>0.25)=%.2f%% "
        "flip_rate=%.2f%%\n",
        name, b.rows, 100.0 * b.top1_agree / n, b.kl_sum / n, mean_omm,
        0.5 * mean_omm, 100.0 * b.w05_gt_025 / n, 100.0 * b.flip_05 / n,
        1.0 * mean_omm, 100.0 * b.w10_gt_025 / n, 100.0 * b.flip_10 / n);
  };
  print_stats("ALL", all_st);
  print_stats("THINKING", think_st);
  print_stats("NON_THINKING", nonthink_st);

  // 3. 1-pass greedy coherence on 3 prompts (prose, code, math).
  eng_off.reset();
  for (int idx : {0, 5, 10}) {
    inkling2::text::PromptTokens pt;
    inkling2::text::RenderOptions ro;
    ro.generation_prefix = inkling2::text::RenderOptions::Prefix::kHeader;
    REQUIRE(
        inkling2::text::RenderAndTokenize(
            *tok, {inkling2::text::Message::Text("user", prompts[idx].text)},
            {}, ro, &pt, &err),
        "render sample");
    const std::vector<std::int32_t> pids(pt.ids.begin(), pt.ids.end());
    auto s = eng_p1->NewSession();
    eng_p1->Forward(*s, pids);
    std::vector<std::uint32_t> gen;
    std::int32_t next = ArgmaxRow(s->logits());
    gen.push_back(static_cast<std::uint32_t>(next));
    while (gen.size() < 160 && !tok->IsStop(static_cast<std::uint32_t>(next))) {
      eng_p1->Step(*s, next, &next);
      gen.push_back(static_cast<std::uint32_t>(next));
    }
    const std::string decoded = tok->Decode(gen);
    std::printf("--- 1-pass sample [%s] %s ---\n%s\n", prompts[idx].cat,
                prompts[idx].text, decoded.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  bool diag = false;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if (k == "--diag") {
      diag = true;
    } else {
      std::fprintf(stderr,
                   "usage: inkling2_loopcd_test [--model GGUF] [--diag]\n");
      return 2;
    }
  }
  if (diag) {
    REQUIRE(!model.empty(), "--diag requires --model GGUF");
    RunWeakModelDiagnostic(model);
    std::printf("ALL PASS\n");
    return 0;
  }
  RunKernelUnitTests();
  if (!model.empty()) {
    RunEngineTests(model);
  } else {
    std::printf("NOTE: --model not given; skipped §6.2 engine checks\n");
  }
  std::printf("ALL PASS\n");
  return 0;
}
