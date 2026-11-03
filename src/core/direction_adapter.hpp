#ifndef GUFO_CORE_DIRECTION_ADAPTER_HPP_
#define GUFO_CORE_DIRECTION_ADAPTER_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

/// Direction adapters edit what chosen sublayers write into the residual
/// stream, leaving the weights unchanged. A GGUF with
/// `general.type = adapter` and `adapter.type = direction`, for the model
/// named by `general.architecture`:
///
///   adapter.direction.mode   string, "project" or "add"
///   adapter.direction.scale  float32, optional (default 1)
///   blk.{layer}.{site}.direction  F32, ne = [width, k]
///
/// At a site, each row x of the sublayer output becomes
///   project: x -= s (x . v) v for each of the k vectors v in turn
///   add:     x += s v for each of the k vectors
/// with s = adapter.direction.scale times the run-time scale. Vectors need
/// not be unit length (their norms carry the strength); `project` vectors
/// should be mutually orthogonal. The sites are the model's own names.
enum class DirectionMode : std::uint8_t { kProject, kAdd };

struct DirectionSet {
  std::uint32_t layer = 0;
  std::uint32_t site = 0;  // index into DirectionAdapterShape::sites
  std::uint32_t count = 0;
  std::vector<float> vectors;  // [count][width]
};

struct DirectionAdapter {
  /// gguf-sha256-v1 digest of the file (the prompt cache keys on it).
  std::string identity;
  DirectionMode mode = DirectionMode::kProject;
  float scale = 1.0f;  // the file's scale; callers multiply theirs in
  std::uint32_t width = 0;
  std::vector<DirectionSet> sets;  // ordered by layer, then site
};

/// What the loading model accepts.
struct DirectionAdapterShape {
  std::string_view architecture;
  std::uint32_t width = 0;
  std::uint32_t layers = 0;
  std::span<const std::string_view> sites;
};

[[nodiscard]] std::string_view ToString(DirectionMode mode) noexcept;

/// Loads and validates a direction adapter for `shape`. Sets whose vectors
/// are all zero are dropped. Returns nullopt with *error_msg set when the
/// file is not a direction adapter for this model.
[[nodiscard]] std::optional<DirectionAdapter> LoadDirectionAdapter(
    const std::filesystem::path& path, const DirectionAdapterShape& shape,
    std::string* error_msg = nullptr);

}  // namespace gufo::core

#endif  // GUFO_CORE_DIRECTION_ADAPTER_HPP_
