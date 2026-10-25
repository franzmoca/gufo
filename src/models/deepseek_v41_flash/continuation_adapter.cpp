#include "src/models/deepseek_v41_flash/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "src/core/hip/snapshot_copy.hpp"
#include "src/models/deepseek_v41_flash/runtime/model_data_internal.h"

namespace gufo::models::deepseek_v41_flash {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
// Append rows share chunks of 2048 rows: rows are small (4-256 bytes), and
// fewer chunks mean fewer copies when lent rows are spilled or captured.
constexpr Rows kChunkRows = 2048;
constexpr std::array<int, 4> kRowParts = {
    DS41_STATE_COMP, DS41_STATE_COMP_SCALE, DS41_STATE_INDEX_K,
    DS41_STATE_INDEX_SCALE};
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{};
};
static_assert(std::is_trivially_copyable_v<Metadata>);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
// Loaded byte ranges cover [0, end) without a gap.
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    ds41_session_set_write_hook(Native(), &State::OnWrite, this);
  }
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    BeforeRelease();
    ds41_session_set_write_hook(Native(), nullptr, nullptr);
  }
  [[nodiscard]] ds41_session* Native() const { return session->session_; }
  [[nodiscard]] Rows Tokens() const {
    return static_cast<Rows>(ds41_session_pos(Native()));
  }
  // An empty session is a valid cold slot; guards read through `guarding`.
  bool IsValid() const noexcept override {
    return !restoring && !failed &&
           (guarding || Tokens() == 0 || ds41_session_valid(Native()));
  }
  void RequireIdle() const {
    if (!IsValid() || pending)
      throw std::logic_error("DeepSeek V4.1 slot is not ready for execution");
  }
  // Runs before an evaluation of positions [first, end) writes any row.
  static void OnWrite(void* data, std::uint32_t first, std::uint32_t end) {
    static_cast<State*>(data)->BeforeWrite(first, end);
  }
  void BeforeWrite(Rows first, Rows end) {
    if (restoring || failed || pending)
      throw std::logic_error("DeepSeek V4.1 slot is not ready for mutation");
    const Guarding readable(*this);
    for (const auto& component : adapter.components_) {
      if (!cache::HasRows(component.descriptor.kind))
        continue;
      // A row is written once its ratio tokens are complete.
      const Rows a = first / component.ratio, b = end / component.ratio;
      if (a < b)
        guard.BeforeOverwrite(component.descriptor.id, a, b);
    }
  }
  void GuardAll() {
    const Guarding readable(*this);
    for (const auto& component : adapter.components_)
      if (cache::HasRows(component.descriptor.kind))
        guard.BeforeOverwrite(component.descriptor.id, 0,
                              adapter.Capacity(component));
  }
  void BeforeRelease() noexcept {
    const Guarding readable(*this);
    for (const auto& component : adapter.components_)
      if (cache::HasRows(component.descriptor.kind))
        guard.BeforeRelease(component.descriptor.id, 0,
                            adapter.Capacity(component));
  }
  struct Guarding {
    explicit Guarding(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~Guarding() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false}, guarding{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  // During a restore: the token buffer, positions and loaded byte ranges.
  int* restore_tokens{};
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  Metadata metadata{};
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      identity_(std::move(identity)) {
  if (!model_ || identity_.empty() || context < 2 ||
      context > model_->MaxContext())
    throw std::invalid_argument("invalid DeepSeek V4.1 adapter configuration");
  const auto add = [&](ContinuationPart part, std::uint32_t layer,
                       cache::ComponentDescriptor d, int state = 0,
                       std::uint32_t ratio = 1) {
    d.id = {static_cast<std::uint32_t>(components_.size() + 1)};
    d.layout_version = 1;
    components_.push_back({d, part, layer, state, ratio});
  };
  add(ContinuationPart::kTokens, 0,
      {.kind = cache::ComponentKind::kAppendRows,
       .row_bytes = sizeof(int),
       .rows_per_chunk = kChunkRows});
  const std::size_t index_dim = DS41_N_INDEXER_HEAD_DIM;
  const std::array<std::size_t, 4> row_bytes = {
      DS41_N_HEAD_DIM / 2, DS41_N_HEAD_DIM / 16, index_dim / 2, index_dim / 32};
  // Every window ring and every compressor frontier form one private
  // component each: cache bookkeeping scales with the component count.
  std::size_t pool_bytes = 0;
  for (std::uint32_t l = 0; l < DS41_N_LAYER; ++l) {
    const auto ratio = ds41_layer_state_ratio(l);
    if (!ratio)
      continue;
    for (std::size_t i = 0; i < kRowParts.size(); ++i)
      add(ContinuationPart::kRows, l,
          {.kind = cache::ComponentKind::kAppendRows,
           .row_bytes = row_bytes[i],
           .rows_per_chunk = kChunkRows},
          kRowParts[i], ratio);
    pool_bytes += 2 * std::size_t{ratio} * DS41_N_HEAD_DIM * sizeof(float);
  }
  add(ContinuationPart::kWindow, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = std::size_t{DS41_N_LAYER} * DS41_N_SWA * DS41_N_HEAD_DIM *
                      sizeof(float)});
  add(ContinuationPart::kPool, 0,
      {.kind = cache::ComponentKind::kPrivateState, .state_bytes = pool_bytes});
  add(ContinuationPart::kLogits, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = std::size_t(model_->VocabSize()) * sizeof(float)});
  add(ContinuationPart::kMetadata, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = sizeof(Metadata)});
  for (const auto& component : components_)
    descriptors_.push_back(component.descriptor);
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Component layout version.
  put(context);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument(
        "slot belongs to another DeepSeek V4.1 adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > components_.size())
    throw std::invalid_argument("invalid DeepSeek V4.1 component ID");
  return components_[id.value - 1];
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
Rows ContinuationAdapter::Capacity(const ContinuationComponent& c) const {
  return c.part == ContinuationPart::kTokens ? context_
                                             : context_ / c.ratio + 1;
}
std::vector<cache::ComponentPosition> ContinuationAdapter::PositionsAt(
    Rows tokens) const {
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& c : components_)
    positions.push_back({c.descriptor.id, tokens / c.ratio});
  return positions;
}
std::vector<ContinuationBackingClass> ContinuationAdapter::BackingPlan(
    std::size_t ram) const {
  // Rows, private states and metadata share the same committed pages; a
  // payload takes a contiguous page group and charges all of its slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(core::SessionMode::kAutoregressive,
                                       context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid DeepSeek V4.1 slot");
  return PositionsAt(state.Tokens());
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid DeepSeek V4.1 prefill plan");
  // Session::Sync evaluates in decoder batches from the frontier.
  const Rows batch = std::max<Rows>(
      ds41_decoder_batch_capacity(ds41_session_decoder(state.Native())), 1);
  std::vector<Rows> result;
  while (first < end) {
    first += std::min(end - first, batch);
    result.push_back(first);
  }
  return result;
}
void* ContinuationAdapter::RowData(State& state,
                                   const ContinuationComponent& c) const {
  if (c.part != ContinuationPart::kRows)
    throw std::invalid_argument("component has no device rows");
  std::uint64_t bytes = 0;
  void* data = ds41_decoder_state_part(ds41_session_decoder(state.Native()),
                                       c.layer, c.state, &bytes);
  if (!data || bytes != c.descriptor.row_bytes)
    throw std::logic_error("DeepSeek V4.1 decoder rows differ from layout");
  return data;
}
std::vector<ContinuationAdapter::Piece> ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& c) const {
  auto* decoder = ds41_session_decoder(state.Native());
  const auto total = c.descriptor.state_bytes;
  std::vector<Piece> pieces;
  std::size_t offset = 0;
  // Appends the layer's part at the next offset.
  const auto part = [&](std::uint32_t layer, int which) {
    std::uint64_t bytes = 0;
    void* data = ds41_decoder_state_part(decoder, layer, which, &bytes);
    if (!data)
      throw std::logic_error("DeepSeek V4.1 decoder lacks a private part");
    pieces.push_back({data, offset, bytes});
    offset += bytes;
  };
  switch (c.part) {
    case ContinuationPart::kWindow:
      for (std::uint32_t l = 0; l < DS41_N_LAYER; ++l)
        part(l, DS41_STATE_WINDOW);
      break;
    case ContinuationPart::kPool:
      for (std::uint32_t l = 0; l < DS41_N_LAYER; ++l) {
        if (!ds41_layer_state_ratio(l))
          continue;
        part(l, DS41_STATE_POOL_KV);
        part(l, DS41_STATE_POOL_SCORE);
      }
      break;
    case ContinuationPart::kLogits:
      pieces.push_back({ds41_session_logits(state.Native()), 0, total, true});
      break;
    case ContinuationPart::kMetadata:
      pieces.push_back({&state.metadata, 0, total, true});
      break;
    default:
      throw std::invalid_argument("component is not private state");
  }
  std::size_t covered = 0;
  for (const auto& piece : pieces)
    covered += piece.bytes;
  if (covered != total)
    throw std::logic_error("DeepSeek V4.1 private state differs from layout");
  return pieces;
}
cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}
void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (state.restoring || state.failed || state.pending)
    throw std::logic_error(
        "reset failed restores and settle transfers before BeginRestore");
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete DeepSeek V4.1 restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& d : descriptors_) {
    if (std::ranges::count(positions, d.id, &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument(
          "missing or duplicate DeepSeek V4.1 frontier");
    ordered.push_back(
        *std::ranges::find(positions, d.id, &cache::ComponentPosition::id));
  }
  const auto n = ordered.front().valid_rows;
  if (!n || n >= context_ || ordered != PositionsAt(n))
    throw std::invalid_argument(
        "inconsistent DeepSeek V4.1 component frontiers");
  // Guard every replaced range while the old rows are still readable.
  state.GuardAll();
  state.restore_tokens =
      ds41_session_begin_restore(state.Native(), static_cast<std::uint32_t>(n));
  if (!state.restore_tokens)
    throw std::runtime_error("DeepSeek V4.1 session cannot begin a restore");
  state.expected = std::move(ordered);
  state.coverage.assign(descriptors_.size(), {});
  state.metadata = {};
  state.restoring = true;
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& c = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid DeepSeek V4.1 slot");
  const auto row = c.descriptor.row_bytes;
  if (!cache::HasRows(c.descriptor.kind) || first > end ||
      end > state.Tokens() / c.ratio || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid DeepSeek V4.1 row capture");
  const std::byte* source = nullptr;
  if (!bytes.empty())
    source = c.part == ContinuationPart::kTokens
                 ? reinterpret_cast<const std::byte*>(
                       ds41_session_tokens(state.Native()))
                 : static_cast<const std::byte*>(RowData(state, c));
  if (source)
    source += first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (bytes.empty())
          return;
        if (c.part == ContinuationPart::kTokens)
          std::memcpy(bytes.data(), source, bytes.size());
        else
          Check(hip::CopySnapshotToHost(bytes.data(), source, bytes.size(),
                                        transfer.Native()));
      });
}
std::span<const std::byte> ContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& c = Component(id);
  // The host token vector can move when execution appends; device rows stay
  // fixed for the session and are guarded before every write.
  if (c.part != ContinuationPart::kRows)
    return {};
  if (!state.IsValid() || first >= end || end > state.Tokens() / c.ratio)
    throw std::invalid_argument("invalid DeepSeek V4.1 borrowed range");
  const auto row = c.descriptor.row_bytes;
  return {static_cast<const std::byte*>(RowData(const_cast<State&>(state), c)) +
              first * row,
          (end - first) * row};
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DeepSeek V4.1 load requires an unfailed restore");
  try {
    const auto& c = Component(id);
    const auto row = c.descriptor.row_bytes;
    if (!cache::HasRows(c.descriptor.kind) || first > end ||
        end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid DeepSeek V4.1 row restore");
    std::byte* destination = nullptr;
    if (!bytes.empty())
      destination = (c.part == ContinuationPart::kTokens
                         ? reinterpret_cast<std::byte*>(state.restore_tokens)
                         : static_cast<std::byte*>(RowData(state, c))) +
                    first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (bytes.empty())
            return;
          if (c.part == ContinuationPart::kTokens)
            std::memcpy(destination, bytes.data(), bytes.size());
          else
            Check(hip::CopySnapshotToDevice(destination, bytes.data(),
                                            bytes.size(), transfer.Native()));
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect DeepSeek V4.1 private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.Tokens() == 0)
    throw std::logic_error(
        "cannot capture an empty or invalid DeepSeek V4.1 slot");
  const auto& c = Component(id);
  const auto total = c.descriptor.state_bytes;
  if (c.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid DeepSeek V4.1 private piece");
  Metadata metadata{.position = static_cast<std::uint32_t>(state.Tokens())};
  auto pieces = PrivateRegion(state, c);
  if (c.part == ContinuationPart::kMetadata)
    pieces = {{&metadata, 0, total, true}};
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        for (const auto& piece : pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          const auto* source = static_cast<const std::byte*>(piece.pointer) +
                               first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes) {
    As(slot).failed = true;
    throw std::invalid_argument("incorrect DeepSeek V4.1 private buffer size");
  }
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DeepSeek V4.1 load requires an unfailed restore");
  try {
    const auto& c = Component(id);
    const auto total = c.descriptor.state_bytes;
    if (c.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid DeepSeek V4.1 private piece");
    const auto pieces = PrivateRegion(state, c);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error(
        "DeepSeek V4.1 validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  const auto n = state.expected.front().valid_rows;
  if (state.metadata.version != 1 || state.metadata.position != n ||
      ds41_session_finish_restore(state.Native()) != 0)
    return refuse();
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    resetting = &state;
    state.BeforeRelease();
    ds41_session_invalidate(state.Native());
    if (ds41_decoder_reset(ds41_session_decoder(state.Native())) != 0)
      throw std::runtime_error("DeepSeek V4.1 decoder reset failed");
    state.failed = false;
    state.restoring = false;
    state.restore_tokens = nullptr;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}
}  // namespace gufo::models::deepseek_v41_flash
