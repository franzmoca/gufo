#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_ADAPTER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/models/deepseek_v41_flash/engine.hpp"

namespace gufo::models::deepseek_v41_flash {

// One continuation's state, by component (ID = index + 1).
enum class ContinuationPart : std::uint8_t {
  kTokens,
  kRows,    // one row array of a KV source layer (`state` part)
  kWindow,  // the sliding-window rings of all layers
  kPool,    // the compressor frontiers of all KV source layers
  kLogits,
  kMetadata,
  kDrafter,  // the DSpark drafter's stage rings and their positions
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
  int state{0};            // ds41_state_part of kRows
  std::uint32_t ratio{1};  // tokens per row of kRows
};
struct ContinuationBackingClass {
  std::size_t block_bytes{}, blocks{}, metadata_bytes{};
};

// Serves DeepSeek V4.1 Flash sessions to the component cache. Tokens and the
// FP4 compressed-KV and index-K rows of each KV source layer are append rows,
// counted per component (one row per compression ratio tokens); the device
// rows can be lent. The 128-row window rings, the compressor frontiers, the
// frontier logits and metadata are one private component each, and so are
// the DSpark drafter's rings for speculative sessions. Sessions evaluate
// synchronously, so no device work is pending once execution returns.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model>, std::uint32_t context,
                      cache::Identity, bool speculative = false);
  ~ContinuationAdapter() override;
  [[nodiscard]] Session& GetSession(cache::Slot&);
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const;
  [[nodiscard]] std::span<const ContinuationComponent> Layout() const {
    return components_;
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows,
      cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  // Valid rows of each component at `tokens` evaluated tokens.
  [[nodiscard]] std::vector<cache::ComponentPosition> PositionsAt(
      cache::Rows tokens) const;
  // Rows a component may hold at most (its allocation).
  [[nodiscard]] cache::Rows Capacity(const ContinuationComponent&) const;
  [[nodiscard]] void* RowData(State&, const ContinuationComponent&) const;
  [[nodiscard]] std::vector<Piece> PrivateRegion(
      State&, const ContinuationComponent&) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(gufo::hip::TransferStream&)>&);
  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  bool speculative_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};
}  // namespace gufo::models::deepseek_v41_flash
#endif
