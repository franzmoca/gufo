#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// Routed expert weight formats with decode and verification kernels.
enum class ExpertFormat : std::uint8_t { kQ4_K, kQ5_K, kQ6_K, kQ8_0, kQ5_1 };

/// Rows whose assignments one expert group can hold: an expert appears at
/// most once per row, so this bounds the rows of a grouped (decode or
/// verification) forward.
inline constexpr std::uint32_t kMaxGroupSlots = 16;
/// Ints per group in the table: expert, slot count, slots.
inline constexpr std::uint32_t kGroupInts = 2 + kMaxGroupSlots;
/// Table ints for up to `max_groups` groups (a leading group count).
[[nodiscard]] constexpr std::size_t ExpertGroupInts(
    std::uint32_t max_groups) noexcept {
  return 1 + std::size_t{max_groups} * kGroupInts;
}
/// Router widths the routing kernel holds in registers.
inline constexpr std::uint32_t kMaxRouterHidden = 3072;

/// Expert routing of `rows` attention residual rows x ([rows][hidden]):
///   logits[r][e] = router[e] . (rms(x[r]) / sqrt(hidden) * router_scale),
/// the `used` largest logits (ties to the lower expert) in descending order
/// in ids[r][j], and weights[r][j] = softmax over the chosen logits times
/// expert_scale[ids[r][j]]. Every row's arithmetic is independent of the
/// batch. Optional outputs: `counts` ([experts], zeroed by the caller)
/// accumulates assignments per expert; `groups` (rows <= kMaxGroupSlots)
/// receives the group table the routed GEMV reads: every selected expert in
/// increasing order with its slots (r * used + j) in increasing order.
struct MoeRouteArgs {
  const float* x;
  const float* router;        ///< F32 [experts][hidden]
  const float* router_scale;  ///< [hidden]
  const float* expert_scale;  ///< [experts]
  float* logits;              ///< scratch, [rows][experts]
  std::int32_t* ids;          ///< [rows][used]
  float* weights;             ///< [rows][used]
  std::uint32_t* counts;
  std::int32_t* groups;
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t experts;
  std::uint32_t used;
  float eps;
};
void MoeRoute(const MoeRouteArgs& args, hipStream_t stream);

/// Grouped routed projection over FP32 activations: for every slot s of
/// every group in `groups` (at most `max_groups`, the launch bound),
/// y[s][0..m) = W[expert] x[s / x_div], where W is [experts][m][k] in
/// `format`. Each expert's weights are decoded once per pass and applied to
/// up to four of its slots; a slot's FMA order depends only on its row and
/// the shape, so decode and verification rows round identically. Returns
/// false without launching for unsupported shapes.
[[nodiscard]] bool LaunchRoutedGemv(ExpertFormat format, const void* w,
                                    const std::int32_t* groups,
                                    std::uint32_t max_groups, const float* x,
                                    std::uint32_t x_div, float* y,
                                    std::uint32_t m, std::uint32_t k,
                                    hipStream_t stream);

/// dense[r] = rms(dense[r]) * norm1 + rms(sum_j weights[r][j] *
/// experts[r * used + j]) * norm2: the layer's feed-forward output from the
/// dense MLP and the expert mixture, summed over j in order.
void MoeCombine(float* dense, const float* experts, const float* weights,
                const float* norm1, const float* norm2, std::uint32_t rows,
                std::uint32_t hidden, std::uint32_t used, float eps,
                hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_HPP_
