#include "src/models/gemma4/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "src/core/quant/ggml_gemm.hpp"

namespace gufo::models::gemma4 {
namespace {

float RoundHalf(float x) {
  // Round-to-nearest-even through binary16, including subnormals.
  return static_cast<float>(static_cast<_Float16>(x));
}

/// y = x * rsqrt(mean(x^2) + eps) * w; `w` null means an unweighted norm.
void RmsNorm(const float* x, const float* w, std::size_t n, float eps,
             float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    const double v = x[i] * scale;
    y[i] = static_cast<float>(w != nullptr ? v * w[i] : v);
  }
}

const float* Vec(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

/// NEOX rotation of one head: pair (i, i + dim/2) turns by
/// pos * theta^(-2i/dim) / factor[i].
void Rope(float* head, std::uint32_t dim, std::uint32_t position, float theta,
          const float* factors) {
  const std::uint32_t half = dim / 2;
  for (std::uint32_t i = 0; i < half; ++i) {
    double angle = static_cast<double>(position) *
                   std::pow(static_cast<double>(theta),
                            -2.0 * static_cast<double>(i) / dim);
    if (factors != nullptr) {
      angle /= factors[i];
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double a = head[i];
    const double b = head[i + half];
    head[i] = static_cast<float>(a * c - b * s);
    head[i + half] = static_cast<float>(a * s + b * c);
  }
}

double GeluTanh(double x) {
  constexpr double kSqrt2OverPi = 0.79788456080286535587989211986876;
  return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

}  // namespace

Reference::Reference(const ModelWeights& weights, Storage storage)
    : weights_(weights), storage_(storage) {
  Reset();
}

void Reference::Reset() {
  position_ = 0;
  keys_.assign(weights_.config.num_layers, {});
  values_.assign(weights_.config.num_layers, {});
  layer_trace_.clear();
}

/// Q8_1 round trip: per 32 values d = amax / 127 (stored as binary16) and
/// q = round(x / d), as ggml's quantize_row_q8_1.
void RoundQ8(const float* x, std::size_t n, float* y) {
  for (std::size_t b = 0; b < n; b += 32) {
    float amax = 0.0F;
    for (std::size_t i = b; i < b + 32; ++i) {
      amax = std::max(amax, std::fabs(x[i]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    const float d_half = RoundHalf(d);
    for (std::size_t i = b; i < b + 32; ++i) {
      y[i] = std::round(x[i] * id) * d_half;
    }
  }
}

void Reference::MatMul(const TensorRef& weight, const float* x_in,
                       std::size_t rows, float* out) const {
  const std::size_t cols = weight.cols;
  std::vector<float> rounded;
  const float* x = x_in;
  if (storage_ == Storage::kQ8Activations &&
      weight.type != core::GgmlType::kF32) {
    rounded.resize(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
      RoundQ8(x_in + r * cols, cols, rounded.data() + r * cols);
    }
    x = rounded.data();
  }
  const std::size_t outputs = weight.rows;
  const std::size_t row_bytes = weight.RowBytes();
#pragma omp parallel
  {
    std::vector<float> w(cols);
#pragma omp for schedule(static)
    for (std::size_t o = 0; o < outputs; ++o) {
      quant::Dequantize(
          weight.type,
          static_cast<const std::uint8_t*>(weight.data) + o * row_bytes,
          w.data(), cols);
      for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        double acc = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
          acc += static_cast<double>(w[c]) * xr[c];
        }
        out[r * outputs + o] = static_cast<float>(acc);
      }
    }
  }
}

void Reference::Attention(std::uint32_t layer, const float* q, std::size_t rows,
                          float* out) const {
  const Config& c = weights_.config;
  const std::uint32_t dim = c.HeadDim(layer);
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t kv_heads = c.kv_heads[layer];
  const std::uint32_t group = heads / kv_heads;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * dim;
  const auto& keys = keys_[layer];
  const auto& values = values_[layer];
  const std::uint32_t first = position_;  // position of row 0
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      const std::uint32_t pos = first + static_cast<std::uint32_t>(r);
      const std::uint32_t lo = c.IsSliding(layer) && pos + 1 > c.sliding_window
                                   ? pos + 1 - c.sliding_window
                                   : 0;
      const float* qh = q + (r * heads + h) * dim;
      const std::uint32_t kvh = h / group;
      std::vector<double> scores(pos + 1 - lo);
      double max_score = -INFINITY;
      for (std::uint32_t p = lo; p <= pos; ++p) {
        const float* k = keys.data() + p * kv_stride + kvh * dim;
        double s = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d) {
          s += static_cast<double>(qh[d]) * k[d];
        }
        scores[p - lo] = s;  // attention scale is 1
        max_score = std::max(max_score, s);
      }
      double total = 0.0;
      for (double& s : scores) {
        s = std::exp(s - max_score);
        total += s;
      }
      std::vector<double> acc(dim, 0.0);
      for (std::uint32_t p = lo; p <= pos; ++p) {
        const float* v = values.data() + p * kv_stride + kvh * dim;
        const double weight = scores[p - lo] / total;
        for (std::uint32_t d = 0; d < dim; ++d) {
          acc[d] += weight * v[d];
        }
      }
      float* o = out + (r * heads + h) * dim;
      for (std::uint32_t d = 0; d < dim; ++d) {
        o[d] = static_cast<float>(acc[d]);
      }
    }
  }
}

void Reference::Forward(std::span<const TokenId> tokens,
                        std::vector<float>* logits,
                        std::vector<float>* hidden) {
  const Config& c = weights_.config;
  const std::size_t n = tokens.size();
  const std::size_t d = c.hidden_size;
  const float eps = c.rms_eps;
  std::vector<float> x(n * d);
  std::vector<float> h(n * d);

  // Scaled token embeddings (the scale is applied in float, as llama.cpp).
  const float embed_scale = std::sqrt(static_cast<float>(d));
  const std::size_t embed_row = weights_.token_embd.RowBytes();
  for (std::size_t r = 0; r < n; ++r) {
    quant::Dequantize(
        weights_.token_embd.type,
        static_cast<const std::uint8_t*>(weights_.token_embd.data) +
            static_cast<std::size_t>(tokens[r]) * embed_row,
        x.data() + r * d, d);
    for (std::size_t i = 0; i < d; ++i) {
      x[r * d + i] *= embed_scale;
    }
  }
  if (trace_) {
    layer_trace_.assign(static_cast<std::size_t>(c.num_layers) * n * d, 0.0F);
  }

  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const LayerWeights& w = weights_.layers[l];
    const std::uint32_t dim = c.HeadDim(l);
    const std::uint32_t heads = c.num_heads;
    const std::uint32_t kv_heads = c.kv_heads[l];
    const std::size_t q_dim = c.QDim(l);
    const std::size_t kv_dim = c.KvDim(l);
    const float* factors =
        c.IsSliding(l) ? nullptr : weights_.rope_factors.values.data();

    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&x[r * d], Vec(w.attn_norm), d, eps, &h[r * d]);
    }
    std::vector<float> q(n * q_dim);
    std::vector<float> k(n * kv_dim);
    std::vector<float> v(n * kv_dim);
    MatMul(w.attn_q, h.data(), n, q.data());
    MatMul(w.attn_k, h.data(), n, k.data());
    if (w.attn_v.empty()) {
      v = k;  // global layers: V is the raw K projection
    } else {
      MatMul(w.attn_v, h.data(), n, v.data());
    }
    for (std::size_t r = 0; r < n; ++r) {
      const auto pos = static_cast<std::uint32_t>(position_ + r);
      for (std::uint32_t hh = 0; hh < heads; ++hh) {
        float* head = &q[(r * heads + hh) * dim];
        RmsNorm(head, Vec(w.attn_q_norm), dim, eps, head);
        Rope(head, dim, pos, c.RopeTheta(l), factors);
      }
      for (std::uint32_t hh = 0; hh < kv_heads; ++hh) {
        float* kh = &k[(r * kv_heads + hh) * dim];
        float* vh = &v[(r * kv_heads + hh) * dim];
        RmsNorm(kh, Vec(w.attn_k_norm), dim, eps, kh);
        Rope(kh, dim, pos, c.RopeTheta(l), factors);
        RmsNorm(vh, nullptr, dim, eps, vh);
      }
    }
    if (storage_ != Storage::kFloat32) {
      for (auto& value : k)
        value = RoundHalf(value);
      for (auto& value : v)
        value = RoundHalf(value);
    }
    keys_[l].insert(keys_[l].end(), k.begin(), k.end());
    values_[l].insert(values_[l].end(), v.begin(), v.end());

    std::vector<float> attn(n * q_dim);
    Attention(l, q.data(), n, attn.data());
    std::vector<float> o(n * d);
    MatMul(w.attn_output, attn.data(), n, o.data());
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&o[r * d], Vec(w.post_attn_norm), d, eps, &o[r * d]);
      for (std::size_t i = 0; i < d; ++i) {
        x[r * d + i] += o[r * d + i];
      }
      RmsNorm(&x[r * d], Vec(w.ffn_norm), d, eps, &h[r * d]);
    }

    const std::size_t ff = c.ffn_size;
    std::vector<float> gate(n * ff);
    std::vector<float> up(n * ff);
    MatMul(w.ffn_gate, h.data(), n, gate.data());
    MatMul(w.ffn_up, h.data(), n, up.data());
    for (std::size_t i = 0; i < n * ff; ++i) {
      gate[i] = static_cast<float>(GeluTanh(gate[i]) * up[i]);
    }
    std::vector<float> f(n * d);
    MatMul(w.ffn_down, gate.data(), n, f.data());
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&f[r * d], Vec(w.post_ffn_norm), d, eps, &f[r * d]);
      for (std::size_t i = 0; i < d; ++i) {
        x[r * d + i] = (x[r * d + i] + f[r * d + i]) * w.output_scale;
      }
    }
    if (trace_) {
      std::copy(x.begin(), x.end(),
                layer_trace_.begin() + static_cast<std::size_t>(l) * n * d);
    }
  }

  for (std::size_t r = 0; r < n; ++r) {
    RmsNorm(&x[r * d], Vec(weights_.output_norm), d, eps, &h[r * d]);
  }
  if (hidden != nullptr) {
    *hidden = h;
  }
  if (logits != nullptr) {
    const std::size_t vocab = weights_.vocab_size;
    logits->assign(n * vocab, 0.0F);
    MatMul(weights_.output, h.data(), n, logits->data());
    const double cap = c.final_logit_softcap;
    if (cap > 0.0) {
      for (float& value : *logits) {
        value = static_cast<float>(cap * std::tanh(value / cap));
      }
    }
  }
  position_ += static_cast<std::uint32_t>(n);
}

}  // namespace gufo::models::gemma4
