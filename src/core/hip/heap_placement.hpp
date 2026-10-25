#ifndef GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
#define GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_

#include <cstdint>

namespace gufo::hip {

/// Keep this much free in the device heap for driver command submission.
inline constexpr std::uint64_t kDispatchReserve = 4ull << 30;

/// Component-cache backing (pinned host pages, which the device heap also
/// counts) may grow until this much is free: below the dispatch reserve, so
/// a cache can work when the weights leave less than it, while keeping room
/// for driver submission and the sessions' lazy buffers.
inline constexpr std::uint64_t kCacheBackingReserve = 1ull << 30;

/// True when `bytes` can be taken from `free_bytes` and still leave
/// `reserve` (kDispatchReserve by default).
[[nodiscard]] inline bool HeapFitsAboveReserve(
    std::uint64_t free_bytes, std::uint64_t bytes,
    std::uint64_t reserve = kDispatchReserve) {
  return bytes <= free_bytes && free_bytes - bytes >= reserve;
}

enum class Placement { kMalloc, kManaged };

[[nodiscard]] inline Placement ChooseWeightPlacement(
    std::uint64_t free_bytes, std::uint64_t total_weight_bytes) {
  return HeapFitsAboveReserve(free_bytes, total_weight_bytes)
             ? Placement::kMalloc
             : Placement::kManaged;
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
