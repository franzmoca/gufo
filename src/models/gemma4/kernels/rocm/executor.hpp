#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/gemma4/kernels/rocm/device_model.hpp"

namespace gufo::models::gemma4::rocm {

/// One session's attention state. Global layers keep every position in a
/// token-major binary16 cache; sliding layers keep a ring of `ring` slots,
/// at least window + the largest forward, so no forward overwrites a key
/// still inside a live window and rollback only moves the frontier.
struct KvCache {
  ~KvCache();
  void* allocation{nullptr};
  std::size_t bytes{0};
  std::uint32_t max_context{0};
  std::uint32_t ring{0};
  std::vector<std::uint16_t*> k;  ///< Per layer, binary16.
  std::vector<std::uint16_t*> v;  ///< Per layer, binary16.
};

/// Runs the Gemma 4 graph on device-resident weights. Scratch is shared by
/// every session; calls are serialized on one stream.
class Executor {
public:
  /// `max_rows` bounds one forward (prefill chunk); `max_logit_rows` bounds
  /// how many rows of one forward may request logits.
  Executor(const DeviceModel& model, std::uint32_t max_rows,
           std::uint32_t max_logit_rows, std::uint32_t max_context);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::size_t ScratchBytes(const Config& config,
                                                std::uint32_t vocab,
                                                std::size_t max_cols,
                                                std::uint32_t max_rows,
                                                std::uint32_t max_logit_rows,
                                                std::uint32_t max_context);
  [[nodiscard]] static std::size_t CacheBytes(const Config& config,
                                              std::uint32_t max_context,
                                              std::uint32_t ring);

  [[nodiscard]] std::unique_ptr<KvCache> CreateCache(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;

  /// Evaluates `tokens` at positions [first_position, +tokens.size()) and
  /// writes softcapped logits of `logit_rows` (indices into `tokens`, in
  /// order) to the device logits buffer. Every earlier position must already
  /// be in `cache`.
  void Forward(KvCache& cache, std::span<const std::int32_t> tokens,
               std::uint32_t first_position,
               std::span<const std::uint32_t> logit_rows);

  /// Copies the logits of the last Forward to the host, [rows][vocab].
  void CopyLogits(std::size_t rows, std::vector<float>* out) const;
  /// Post-norm hidden rows of the last Forward, [tokens][hidden].
  [[nodiscard]] const float* hidden() const noexcept { return h_; }
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::uint32_t ring() const noexcept { return ring_; }
  [[nodiscard]] const DeviceModel& model() const noexcept { return model_; }

private:
  /// y[rows][w.rows] = x[rows][w.cols] * W^T; `xq` is the Q8_1 encoding of
  /// x when rows exceed the small-batch limit.
  void Project(const DeviceTensor& w, const float* x, const void* xq,
               std::uint32_t rows, float* y);
  const void* Quantize(const float* x, std::uint32_t rows, std::uint32_t cols);

  const DeviceModel& model_;
  std::uint32_t max_rows_;
  std::uint32_t max_logit_rows_;
  std::uint32_t max_context_;
  std::uint32_t ring_;
  hipStream_t stream_{nullptr};
  void* scratch_{nullptr};
  std::int32_t* tokens_{nullptr};
  std::uint32_t* logit_index_{nullptr};
  float* x_{nullptr};
  float* h_{nullptr};
  float* q_{nullptr};
  float* k_{nullptr};
  float* v_{nullptr};
  float* attn_{nullptr};
  float* o_{nullptr};
  float* gate_{nullptr};
  float* up_{nullptr};
  float* hsel_{nullptr};
  float* logits_{nullptr};
  float* partials_{nullptr};
  void* q8_{nullptr};
};

/// Ring slots for sliding layers given the largest forward.
[[nodiscard]] std::uint32_t RingSlots(const Config& config,
                                      std::uint32_t max_rows) noexcept;

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_
