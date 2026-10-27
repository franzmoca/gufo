/* SSD-backed routed experts. The stage buffers preserve the original expert
 * indices, so decode, prefill and speculative verification use the same
 * kernels and arithmetic as resident weights. */
#include "expert_store.h"

#include <errno.h>
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "model_data_internal.h"
#include "src/cli/serve/logging.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/platform/os_info.hpp"

namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kReserve = 8 * kGiB;
constexpr uint64_t kChunkBytes = 512ull << 20;
constexpr uint64_t kReadBytes = 32ull << 20;
constexpr uint64_t kDirectAlign = 4096;
constexpr uint32_t kReadWorkers = 4;
constexpr uint32_t kRecentRows = 16;
/* Route rows of the largest split evaluation (decode, DSpark verification
 * and joint rows); larger batches read their experts in one pass. */
constexpr uint32_t kSplitRows = DS41_DECODER_MAX_SPECULATIVE * DS41_N_EXPERT_USED;
constexpr uint32_t kSeedPerLayer = DS41_N_EXPERT;
constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();
constexpr std::array<uint16_t, 10240> kDefaultExpertRank = {
#include "expert_rank.inc"
};
constexpr bool valid_default_rank() {
  std::array<bool, DS41_N_LAYER * DS41_N_EXPERT> seen{};
  for (uint32_t index : kDefaultExpertRank) {
    if (index >= seen.size() || seen[index])
      return false;
    seen[index] = true;
  }
  return true;
}
static_assert(valid_default_rank());

struct Entry {
  uint32_t slot = kNoSlot;
  uint64_t last = 0;
  double score = 0;
};

struct Slot {
  uint8_t* data = nullptr;
  uint32_t layer = kNoSlot;
  uint32_t expert = kNoSlot;
};

struct Source {
  int fd = -1;
  int direct_fd = -1;
  uint64_t offset = 0;
  uint64_t bytes = 0;  // one expert
};

struct ReadTask {
  int fd;
  int direct_fd;
  uint64_t offset;
  uint64_t bytes;
  uint8_t* destination;
};

bool read_all(int fd, uint8_t* dst, uint64_t bytes, uint64_t offset) {
  while (bytes) {
    const ssize_t got =
        pread(fd, dst, static_cast<size_t>(bytes), static_cast<off_t>(offset));
    if (got < 0 && errno == EINTR)
      continue;
    if (got <= 0)
      return false;
    dst += got;
    offset += static_cast<uint64_t>(got);
    bytes -= static_cast<uint64_t>(got);
  }
  return true;
}

/* O_DIRECT needs aligned buffers and ranges. The last read can end at an
 * unaligned EOF, provided it contains the entire requested tensor span. */
bool read_direct(int fd, uint8_t* dst, uint64_t bytes, uint64_t offset,
                 uint64_t* skip) {
  if (fd < 0)
    return false;
  const uint64_t aligned = offset / kDirectAlign * kDirectAlign;
  *skip = offset - aligned;
  const uint64_t need = *skip + bytes;
  const uint64_t length =
      (need + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
  uint64_t done = 0;
  while (done < need) {
    const ssize_t got =
        pread(fd, dst + done, static_cast<size_t>(length - done),
              static_cast<off_t>(aligned + done));
    if (got < 0 && errno == EINTR)
      continue;
    if (got <= 0)
      return false;
    done += static_cast<uint64_t>(got);
    if (done < need && static_cast<uint64_t>(got) % kDirectAlign)
      return false;
  }
  return true;
}

double decayed(const Entry& entry, uint64_t now) {
  return entry.score / (1.0 + static_cast<double>(now - entry.last) / 1600.0);
}

/* The queue is sorted before dispatch. Workers read independent ranges into
 * their own pinned buffers and upload on their own streams, so the copies
 * overlap the legacy stream's kernels instead of queueing behind them; each
 * upload is awaited before its buffer is reused. No work outlives Execute, so
 * stage buffers and shard descriptors stay valid. */
class ReadPool {
public:
  ReadPool(const std::array<uint8_t*, kReadWorkers>& buffers, int device)
      : buffers_(buffers), device_(device) {
    bool streams_ok = true;
    for (uint32_t i = 0; i < kReadWorkers && streams_ok; ++i)
      streams_ok = hipStreamCreateWithFlags(&streams_[i], hipStreamNonBlocking) ==
                       hipSuccess &&
                   hipEventCreateWithFlags(&events_[i], hipEventDisableTiming) ==
                       hipSuccess;
    if (!streams_ok) {
      ReleaseResources();
      throw std::runtime_error("ds41 read pool streams");
    }
    try {
      for (uint32_t i = 0; i < kReadWorkers; ++i)
        threads_.emplace_back([this, i] { Work(i); });
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~ReadPool() {
    Stop();
    ReleaseResources();
  }

  bool Execute(const std::vector<ReadTask>& tasks,
               bool (*overlap)(void*, const ds41_expert_overlap*),
               void* overlap_context, const ds41_expert_overlap& info) {
    if (tasks.empty())
      return !overlap || overlap(overlap_context, &info);
    std::unique_lock lock(mutex_);
    tasks_ = &tasks;
    next_.store(0);
    failed_.store(false);
    completed_ = 0;
    ++generation_;
    start_.notify_all();
    lock.unlock();
    const bool overlap_ok = !overlap || overlap(overlap_context, &info);
    lock.lock();
    done_.wait(lock, [this] { return completed_ == threads_.size(); });
    tasks_ = nullptr;
    return overlap_ok && !failed_.load();
  }

  uint64_t direct_reads() const { return direct_reads_.load(); }
  uint64_t buffered_reads() const { return buffered_reads_.load(); }

private:
  void ReleaseResources() {
    for (uint32_t i = 0; i < kReadWorkers; ++i) {
      if (events_[i])
        (void)hipEventDestroy(events_[i]);
      if (streams_[i])
        (void)hipStreamDestroy(streams_[i]);
      events_[i] = nullptr;
      streams_[i] = nullptr;
    }
  }
  void Stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    start_.notify_all();
    for (std::thread& thread : threads_)
      if (thread.joinable())
        thread.join();
  }

  void Work(uint32_t index) {
    const bool device_ok = hipSetDevice(device_) == hipSuccess;
    uint64_t seen = 0;
    for (;;) {
      const std::vector<ReadTask>* tasks = nullptr;
      {
        std::unique_lock lock(mutex_);
        start_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_)
          return;
        seen = generation_;
        tasks = tasks_;
      }
      if (!device_ok)
        failed_.store(true);
      for (;;) {
        const size_t next = next_.fetch_add(1);
        if (next >= tasks->size())
          break;
        const ReadTask& task = (*tasks)[next];
        const uintptr_t raw = reinterpret_cast<uintptr_t>(buffers_[index]);
        auto* buffer = reinterpret_cast<uint8_t*>((raw + kDirectAlign - 1) /
                                                  kDirectAlign * kDirectAlign);
        uint64_t skip = 0;
        const bool direct =
            device_ok &&
            read_direct(task.direct_fd, buffer, task.bytes, task.offset, &skip);
        if (direct)
          ++direct_reads_;
        else {
          skip = 0;
          ++buffered_reads_;
        }
        if (!device_ok ||
            (!direct && !read_all(task.fd, buffer, task.bytes, task.offset)) ||
            hipMemcpyAsync(task.destination, buffer + skip, task.bytes,
                           hipMemcpyHostToDevice, streams_[index]) != hipSuccess ||
            hipEventRecord(events_[index], streams_[index]) != hipSuccess ||
            hipEventSynchronize(events_[index]) != hipSuccess)
          failed_.store(true);
      }
      {
        std::lock_guard lock(mutex_);
        ++completed_;
        if (completed_ == threads_.size())
          done_.notify_one();
      }
    }
  }

  std::array<uint8_t*, kReadWorkers> buffers_;
  int device_;
  std::vector<std::thread> threads_;
  std::array<hipStream_t, kReadWorkers> streams_{};
  std::array<hipEvent_t, kReadWorkers> events_{};
  std::mutex mutex_;
  std::condition_variable start_, done_;
  const std::vector<ReadTask>* tasks_ = nullptr;
  std::atomic<size_t> next_{0};
  std::atomic<bool> failed_{false};
  std::atomic<uint64_t> direct_reads_{0}, buffered_reads_{0};
  uint64_t generation_ = 0;
  size_t completed_ = 0;
  bool stopping_ = false;
};

}  // namespace

uint64_t ds41_expert_available_memory() {
  uint64_t available = gufo::platform::AvailableMemoryBytes();
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  uint64_t cma_free = 0;
  while (std::getline(meminfo, line)) {
    unsigned long long kib = 0;
    if (sscanf(line.c_str(), "MemAvailable: %llu kB", &kib) == 1)
      available = kib * 1024;
    if (sscanf(line.c_str(), "CmaFree: %llu kB", &kib) == 1)
      cma_free = kib * 1024;
  }
  available = available > cma_free ? available - cma_free : 0;
  // Respect cgroup limits as well as physical RAM. Inactive clean file cache
  // can be reclaimed under pressure, just as it can on the host.
  std::ifstream membership("/proc/self/cgroup");
  while (std::getline(membership, line)) {
    if (!line.starts_with("0::/"))
      continue;
    const std::filesystem::path root = "/sys/fs/cgroup";
    const std::filesystem::path relative = line.substr(4);
    bool valid = true;
    for (const auto& part : relative)
      if (part == "..")
        valid = false;
    if (!valid)
      break;
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max"),
          used_file(path / "memory.current");
      uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used)) {
        std::ifstream stat_file(path / "memory.stat");
        uint64_t inactive = 0, dirty = 0, value = 0;
        bool found = false;
        for (std::string key; stat_file >> key >> value;) {
          if (key == "inactive_file") {
            inactive = value;
            found = true;
          } else if (key == "file_dirty" || key == "file_writeback")
            dirty += value;
        }
        if (found)
          used -= std::min(used, inactive - std::min(inactive, dirty));
        available = std::min(available, limit > used ? limit - used : 0);
      }
      if (path == root)
        break;
    }
    break;
  }
  return available;
}

struct ds41_expert_store {
  std::array<std::array<Source, 3>, DS41_N_LAYER> source{};
  std::array<std::array<Entry, DS41_N_EXPERT>, DS41_N_LAYER> entries{};
  std::array<uint8_t*, 3> stage{};
  std::vector<void*> chunks;
  /* Per-expert device weight bases (slot data or stage rows) with pinned
   * host mirrors, and the hit/miss route rows of the current prepare call. */
  std::array<const uint8_t**, 3> table_host{};
  std::array<const uint8_t**, 3> table{};
  int32_t* split_hit = nullptr;
  int32_t* split_miss = nullptr;
  std::array<int, DS41_MAX_SHARDS> direct_fds = [] {
    std::array<int, DS41_MAX_SHARDS> fds;
    fds.fill(-1);
    return fds;
  }();
  std::vector<Slot> slots;
  std::vector<uint32_t> free_slots;
  std::array<uint8_t*, kReadWorkers> read_buffers{};
  std::unique_ptr<ReadPool> readers;
  uint64_t slot_bytes = 0;
  uint64_t cache_limit = 0;
  uint64_t tick = 0;
  uint64_t last_grow_failure = 0;
  uint64_t logged_tokens = 0;
  ds41_expert_store_stats stats;
  ds41_expert_store_stats logged_stats;

  ~ds41_expert_store() {
    readers.reset();
    for (int fd : direct_fds)
      if (fd >= 0)
        (void)close(fd);
    (void)hipDeviceSynchronize();
    for (void* chunk : chunks)
      (void)hipFree(chunk);
    for (void* p : stage)
      if (p)
        (void)hipFree(p);
    for (void* p : read_buffers)
      if (p)
        (void)hipHostFree(p);
    for (uint32_t p = 0; p < 3; ++p) {
      if (table[p])
        (void)hipFree(table[p]);
      if (table_host[p])
        (void)hipHostFree(table_host[p]);
    }
    if (split_hit)
      (void)hipFree(split_hit);
    if (split_miss)
      (void)hipFree(split_miss);
  }

  bool grow() {
    const uint64_t free_bytes = ds41_expert_available_memory();
    if (free_bytes <= kReserve)
      return false;
    uint64_t available = free_bytes - kReserve;
    if (stats.cache_bytes >= cache_limit)
      return false;
    available = std::min(available, cache_limit - stats.cache_bytes);
    const uint64_t count = std::min(available, kChunkBytes) / slot_bytes;
    if (!count)
      return false;
    const uint64_t bytes = count * slot_bytes;
    void* allocation = nullptr;
    if (gufo::hip::AllocateManaged(&allocation, bytes) != hipSuccess)
      return false;
    chunks.push_back(allocation);
    stats.cache_bytes += bytes;
    for (uint64_t i = 0; i < count; ++i) {
      free_slots.push_back(static_cast<uint32_t>(slots.size()));
      slots.push_back({static_cast<uint8_t*>(allocation) + i * slot_bytes,
                       kNoSlot, kNoSlot});
    }
    return true;
  }

  uint32_t acquire(uint32_t layer, uint32_t expert, bool allow_evict) {
    if (free_slots.empty() && last_grow_failure != tick && !grow())
      last_grow_failure = tick;
    if (!free_slots.empty()) {
      const uint32_t slot = free_slots.back();
      free_slots.pop_back();
      return slot;
    }
    if (!allow_evict)
      return kNoSlot;
    uint32_t victim = kNoSlot;
    double lowest = std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < slots.size(); ++i) {
      const Slot& s = slots[i];
      if (s.layer == kNoSlot)
        continue;
      const double score = decayed(entries[s.layer][s.expert], tick);
      if (score < lowest) {
        lowest = score;
        victim = i;
      }
    }
    if (victim == kNoSlot || decayed(entries[layer][expert], tick) < lowest)
      return kNoSlot;
    Slot& s = slots[victim];
    entries[s.layer][s.expert].slot = kNoSlot;
    return victim;
  }
};

ds41_expert_store* ds41_expert_store_create(const ds41_model* model,
                                            const ds41_weights* weights,
                                            uint64_t cache_limit_bytes) {
  if (!model || !weights || model->n_shards > DS41_MAX_SHARDS)
    return nullptr;
  auto owner =
      std::unique_ptr<ds41_expert_store>(new (std::nothrow) ds41_expert_store);
  auto* store = owner.get();
  if (!store)
    return nullptr;
  for (uint32_t s = 0; s < model->n_shards; ++s) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/fd/%d", model->shard_fd[s]);
    store->direct_fds[s] = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
  }
  std::array<uint64_t, 3> max_bytes{};
  for (uint32_t layer = 0; layer < DS41_N_LAYER; ++layer) {
    const auto& w = weights->layer[layer];
    const ds41_tensor* tensors[] = {w.ffn_gate_exps, w.ffn_up_exps,
                                    w.ffn_down_exps};
    uint64_t combined = 0;
    for (uint32_t p = 0; p < 3; ++p) {
      const ds41_tensor* t = tensors[p];
      if (!t || !t->bytes || t->bytes % DS41_N_EXPERT) {
        return nullptr;
      }
      Source& src = store->source[layer][p];
      src.bytes = t->bytes / DS41_N_EXPERT;
      max_bytes[p] = std::max(max_bytes[p], src.bytes);
      combined += src.bytes;
      for (uint32_t s = 0; s < model->n_shards; ++s) {
        const uint64_t start = model->shard_offset[s];
        if (t->abs_offset >= start &&
            t->abs_offset - start <= model->shard_size[s] &&
            t->bytes <= model->shard_size[s] - (t->abs_offset - start)) {
          src.fd = model->shard_fd[s];
          src.direct_fd = store->direct_fds[s];
          src.offset = t->abs_offset - start;
          break;
        }
      }
      if (src.fd < 0) {
        return nullptr;
      }
    }
    store->slot_bytes = std::max(store->slot_bytes, combined);
  }
  store->slot_bytes = (store->slot_bytes + 255) / 256 * 256;
  bool ok = true;
  for (uint32_t i = 0; i < kReadWorkers && ok; ++i)
    ok = hipHostMalloc(&store->read_buffers[i],
                       kReadBytes + 2 * kDirectAlign) == hipSuccess;
  for (uint32_t p = 0; p < 3 && ok; ++p)
    ok =
        hipMalloc(&store->stage[p], max_bytes[p] * DS41_N_EXPERT) == hipSuccess;
  for (uint32_t p = 0; p < 3 && ok; ++p)
    ok = hipHostMalloc(reinterpret_cast<void**>(&store->table_host[p]),
                       DS41_N_EXPERT * sizeof(const uint8_t*)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&store->table[p]),
                   DS41_N_EXPERT * sizeof(const uint8_t*)) == hipSuccess;
  if (ok)
    ok = hipMalloc(reinterpret_cast<void**>(&store->split_hit),
                   kSplitRows * sizeof(int32_t)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&store->split_miss),
                   kSplitRows * sizeof(int32_t)) == hipSuccess;
  if (!ok) {
    fprintf(stderr,
            "ds41: unable to allocate streamed expert staging buffers\n");
    return nullptr;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    return nullptr;
  }
  try {
    store->readers = std::make_unique<ReadPool>(store->read_buffers, device);
  } catch (const std::exception&) {
    return nullptr;
  }
  const uint64_t available = ds41_expert_available_memory();
  store->cache_limit = available > kReserve ? available - kReserve : 0;
  if (cache_limit_bytes)
    store->cache_limit = std::min(store->cache_limit, cache_limit_bytes);
  return owner.release();
}

void ds41_expert_store_destroy(ds41_expert_store* store) {
  delete store;
}

namespace {

struct SeedItem {
  uint32_t layer;
  uint32_t expert;
  uint64_t count;
};

bool seed_ranked(ds41_expert_store* store, const std::vector<SeedItem>& ranked,
                 const char* source) {
  const uint64_t available = ds41_expert_available_memory();
  const uint64_t free_budget = available > kReserve ? available - kReserve : 0;
  const uint64_t slots = std::min<uint64_t>(
      DS41_N_LAYER * DS41_N_EXPERT,
      std::min(store->cache_limit, free_budget) / store->slot_bytes);
  // Leave capacity for session allocations and for experts learned from the
  // actual request. At startup the slots are otherwise all free.
  const size_t target =
      std::min<size_t>({ranked.size(), kDefaultExpertRank.size(),
                        static_cast<size_t>(slots * 15 / 16)});
  if (!target)
    return true;
  int32_t* device_ids = nullptr;
  constexpr uint32_t kSeedSlots = (kSeedPerLayer + DS41_N_EXPERT_USED - 1) /
                                  DS41_N_EXPERT_USED * DS41_N_EXPERT_USED;
  if (hipMalloc(&device_ids, kSeedSlots * sizeof(int32_t)) != hipSuccess)
    return false;
  const auto begin = std::chrono::steady_clock::now();
  const ds41_expert_store_stats previous = store->stats;
  std::array<std::array<bool, DS41_N_EXPERT>, DS41_N_LAYER> selected{};
  std::array<std::vector<SeedItem>, DS41_N_LAYER> by_layer;
  for (size_t i = 0; i < target; ++i)
    by_layer[ranked[i].layer].push_back(ranked[i]);
  bool ok = true;
  for (uint32_t l = 0; l < DS41_N_LAYER && ok; ++l) {
    std::vector<uint32_t> ids;
    for (const SeedItem& item : by_layer[l]) {
      ids.push_back(item.expert);
      store->entries[l][item.expert].score =
          std::min(8.0, std::log2(1.0 + item.count));
      selected[l][item.expert] = true;
    }
    if (ids.empty())
      continue;
    if (ds41_expert_available_memory() <= kReserve)
      break;
    while (ids.size() % DS41_N_EXPERT_USED)
      ids.push_back(ids.back());
    std::vector<int32_t> routes(ids.begin(), ids.end());
    const void *gate = nullptr, *up = nullptr, *down = nullptr;
    ok = hipMemcpy(device_ids, routes.data(), routes.size() * sizeof(int32_t),
                   hipMemcpyHostToDevice) == hipSuccess &&
         ds41_expert_store_prepare(
             store, l, device_ids,
             static_cast<uint32_t>(routes.size() / DS41_N_EXPERT_USED), &gate,
             &up, &down);
  }
  (void)hipFree(device_ids);
  const uint64_t bytes = store->stats.bytes_read - previous.bytes_read;
  const uint64_t allocated = store->stats.cache_bytes;
  store->stats = previous;
  store->stats.cache_bytes = allocated;
  if (ok) {
    uint32_t resident = 0;
    for (uint32_t l = 0; l < DS41_N_LAYER; ++l)
      for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
        resident += selected[l][e] && store->entries[l][e].slot != kNoSlot;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, "ds41",
        "expert seed (%s): cached %u of %zu ranked experts from %.2f GiB in "
        "%.2f s",
        source, resident, target, static_cast<double>(bytes) / kGiB, seconds);
  }
  return ok;
}

}  // namespace

bool ds41_expert_store_seed_default(ds41_expert_store* store) {
  if (!store)
    return false;
  std::vector<SeedItem> ranked;
  ranked.reserve(kDefaultExpertRank.size());
  for (size_t i = 0; i < kDefaultExpertRank.size(); ++i) {
    const uint32_t index = kDefaultExpertRank[i];
    ranked.push_back({index / DS41_N_EXPERT, index % DS41_N_EXPERT,
                      static_cast<uint64_t>(kDefaultExpertRank.size() - i)});
  }
  return seed_ranked(store, ranked, "built-in V4.1 ranking");
}

bool ds41_expert_store_seed(ds41_expert_store* store, const char* path) {
  if (!store || !path)
    return false;
  std::ifstream in(path);
  std::string magic;
  if (!std::getline(in, magic) || magic != "DS41_EXPERT_HISTOGRAM_V1")
    return false;
  std::array<std::array<bool, DS41_N_EXPERT>, DS41_N_LAYER> seen{};
  std::vector<SeedItem> ranked;
  std::string line;
  while (std::getline(in, line)) {
    uint32_t layer = 0, expert = 0;
    uint64_t count = 0;
    std::string extra;
    std::istringstream row(line);
    if (!(row >> layer >> expert >> count) || (row >> extra))
      return false;
    if (layer >= DS41_N_LAYER || expert >= DS41_N_EXPERT || !count ||
        seen[layer][expert])
      return false;
    seen[layer][expert] = true;
    ranked.push_back({layer, expert, count});
  }
  if (!in.eof())
    return false;
  std::sort(
      ranked.begin(), ranked.end(), [](const SeedItem& a, const SeedItem& b) {
        if (a.count != b.count)
          return a.count > b.count;
        return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
      });
  return seed_ranked(store, ranked, path);
}

bool ds41_expert_store_prepare(ds41_expert_store* store, uint32_t layer,
                               const int32_t* routes, uint32_t n_tokens,
                               const void** gate, const void** up,
                               const void** down,
                               bool (*overlap)(void*, const ds41_expert_overlap*),
                               void* overlap_context) {
  if (!store || layer >= DS41_N_LAYER || !routes || !n_tokens || !gate || !up ||
      !down)
    return false;
  const auto start = std::chrono::steady_clock::now();
  const size_t n_routes = static_cast<size_t>(n_tokens) * DS41_N_EXPERT_USED;
  std::vector<int32_t> ids(n_routes);
  if (hipMemcpy(ids.data(), routes, n_routes * sizeof(int32_t),
                hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  std::array<uint32_t, DS41_N_EXPERT> counts{}, recent{};
  for (size_t i = 0; i < ids.size(); ++i) {
    const int32_t id = ids[i];
    if (id < 0 || id >= static_cast<int32_t>(DS41_N_EXPERT)) {
      fprintf(stderr, "ds41: invalid routed expert %d\n", id);
      return false;
    }
    ++counts[id];
    if (i >= (n_tokens - std::min(n_tokens, kRecentRows)) * DS41_N_EXPERT_USED)
      ++recent[id];
  }
  ++store->tick;
  if (layer == 0)
    store->stats.tokens += n_tokens;
  std::array<bool, DS41_N_EXPERT> missing{}, admit{};
  for (uint32_t e = 0; e < DS41_N_EXPERT; ++e) {
    if (!counts[e])
      continue;
    Entry& entry = store->entries[layer][e];
    store->stats.uses += counts[e];
    if (entry.slot != kNoSlot)
      store->stats.hits += counts[e];
    missing[e] = entry.slot == kNoSlot;
    entry.score = decayed(entry, store->tick) + std::min(counts[e], 4u) +
                  (recent[e] ? 4.0 : 0.0);
    entry.last = store->tick;
    // A large prefill scans most experts. Admit its final rows and unusually
    // frequent experts, rather than displacing the decode working set.
    admit[e] = n_tokens <= kRecentRows || recent[e] ||
               counts[e] >= std::max(4u, n_tokens / 32);
  }
  /* Map every expert of the layer to its weights: the cache slot when
   * resident, else the stage row the read fills. The kernels index the
   * table, so resident experts are read in place without a staging copy. */
  const auto& src = store->source[layer];
  for (uint32_t e = 0; e < DS41_N_EXPERT; ++e) {
    const uint32_t slot_id = store->entries[layer][e].slot;
    const uint8_t* base =
        slot_id != kNoSlot ? store->slots[slot_id].data : nullptr;
    uint64_t staged = 0;
    for (uint32_t p = 0; p < 3; ++p) {
      store->table_host[p][e] =
          base ? base + staged : store->stage[p] + (uint64_t)e * src[p].bytes;
      staged += src[p].bytes;
    }
  }
  for (uint32_t p = 0; p < 3; ++p)
    if (hipMemcpy(store->table[p], store->table_host[p],
                  DS41_N_EXPERT * sizeof(uint8_t*),
                  hipMemcpyHostToDevice) != hipSuccess)
      return false;
  /* The hit and miss rows keep their route positions, each array holding -1
   * for the other's experts, so one launch over either array leaves exactly
   * the other launch's rows to fill. */
  ds41_expert_overlap info{};
  info.layer = layer;
  info.rows = static_cast<uint32_t>(n_routes);
  for (uint32_t p = 0; p < 3; ++p)
    info.table[p] = store->table[p];
  if (n_routes <= kSplitRows) {
    std::vector<int32_t> hit(n_routes), miss(n_routes);
    uint32_t miss_rows = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      const int32_t id = ids[i];
      const bool is_miss = missing[id];
      miss_rows += is_miss;
      hit[i] = is_miss ? -1 : id;
      miss[i] = is_miss ? id : -1;
    }
    info.miss_rows = miss_rows;
    info.hit_ids = store->split_hit;
    info.miss_ids = store->split_miss;
    if (hipMemcpy(store->split_hit, hit.data(), n_routes * sizeof(int32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(store->split_miss, miss.data(), n_routes * sizeof(int32_t),
                  hipMemcpyHostToDevice) != hipSuccess)
      return false;
  }
  // Sort projection reads by shard and offset, then merge consecutive misses.
  std::array<uint32_t, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return src[a].fd == src[b].fd ? src[a].offset < src[b].offset
                                  : src[a].fd < src[b].fd;
  });
  std::vector<ReadTask> reads;
  for (uint32_t p : order) {
    for (uint32_t e = 0; e < DS41_N_EXPERT;) {
      if (!missing[e]) {
        ++e;
        continue;
      }
      const uint32_t first = e++;
      while (e < DS41_N_EXPERT && missing[e])
        ++e;
      uint64_t offset = src[p].offset + first * src[p].bytes;
      uint64_t left = (e - first) * src[p].bytes;
      uint8_t* dst = store->stage[p] + first * src[p].bytes;
      while (left) {
        const uint64_t bytes = std::min(left, kReadBytes);
        reads.push_back({src[p].fd, src[p].direct_fd, offset, bytes, dst});
        offset += bytes;
        dst += bytes;
        left -= bytes;
      }
    }
  }
  if (!store->readers->Execute(reads, overlap, overlap_context, info)) {
    fprintf(stderr, "ds41: failed to stage experts or run overlapped work\n");
    return false;
  }
  for (const ReadTask& read : reads)
    store->stats.bytes_read += read.bytes;
  for (uint32_t e = 0; e < DS41_N_EXPERT; ++e) {
    if (!missing[e])
      continue;
    // Fill unused capacity during a scan. Once full, only promoted prefill
    // experts may displace the existing working set.
    const uint32_t slot_id = store->acquire(layer, e, admit[e]);
    if (slot_id == kNoSlot)
      continue;
    Slot& slot = store->slots[slot_id];
    uint8_t* dst = slot.data;
    for (uint32_t p = 0; p < 3; ++p) {
      if (hipMemcpyAsync(dst, store->stage[p] + e * src[p].bytes, src[p].bytes,
                         hipMemcpyDeviceToDevice, nullptr) != hipSuccess) {
        slot.layer = slot.expert = kNoSlot;
        store->free_slots.push_back(slot_id);
        return false;
      }
      dst += src[p].bytes;
    }
    slot.layer = layer;
    slot.expert = e;
    store->entries[layer][e].slot = slot_id;
  }
  *gate = store->table[0];
  *up = store->table[1];
  *down = store->table[2];
  store->stats.stall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
  if (layer == DS41_N_LAYER - 1 &&
      store->stats.tokens >= store->logged_tokens + 64) {
    const ds41_expert_store_stats s = store->stats;
    const ds41_expert_store_stats old = store->logged_stats;
    const uint64_t uses = s.uses - old.uses, hits = s.hits - old.hits;
    const uint64_t tokens = s.tokens - old.tokens;
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, "ds41",
        "expert cache interval: %.1f%% route hits, %.2f MiB read/token, "
        "%.2f ms host wait/token, %.2f GiB allocated",
        uses ? 100.0 * static_cast<double>(hits) / uses : 0.0,
        tokens ? static_cast<double>(s.bytes_read - old.bytes_read) / tokens /
                     (1u << 20)
               : 0.0,
        tokens ? static_cast<double>(s.stall_ns - old.stall_ns) / tokens / 1e6
               : 0.0,
        static_cast<double>(s.cache_bytes) / kGiB);
    store->logged_stats = s;
    store->logged_tokens = store->stats.tokens;
  }
  return true;
}

ds41_expert_store_stats ds41_expert_store_get_stats(
    const ds41_expert_store* store) {
  return store ? store->stats : ds41_expert_store_stats{};
}

void ds41_expert_store_log(const ds41_expert_store* store) {
  if (!store || !store->stats.uses)
    return;
  const auto& s = store->stats;
  gufo::server::Logger::LogFormatted(
      gufo::server::LogLevel::kInfo, "ds41",
      "expert cache total: %.1f%% route hits, %.2f MiB read/token, %.2f ms "
      "host wait/token, %.2f GiB allocated",
      100.0 * static_cast<double>(s.hits) / s.uses,
      s.tokens ? static_cast<double>(s.bytes_read) / s.tokens / (1u << 20)
               : 0.0,
      s.tokens ? static_cast<double>(s.stall_ns) / s.tokens / 1e6 : 0.0,
      static_cast<double>(s.cache_bytes) / kGiB);
  gufo::server::Logger::LogFormatted(
      gufo::server::LogLevel::kInfo, "ds41",
      "expert reads: %llu direct, %llu buffered",
      static_cast<unsigned long long>(store->readers->direct_reads()),
      static_cast<unsigned long long>(store->readers->buffered_reads()));
}
