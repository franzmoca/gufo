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
/* Headroom kept above the cache for everything allocated after it: the
 * shared batch scratch (~1.4 GiB at 1M context), one session's FP4 state
 * (~0.9 GiB at 1M, far less at serving depths) and transient prefill peaks.
 * The resident path keeps a 1 GiB dispatch reserve under the same loads;
 * sessions that no longer fit fall back to the disk prompt cache. */
constexpr uint64_t kReserve = 4 * kGiB;
/* Rows of the compact prefetch stage: two parity halves, so the job a layer
 * adopts and the job submitted for the layer after never share rows. A
 * whole-layer set measured 3.6 GiB; jobs touch only the predicted experts
 * (~70 at pp2048, 6 at decode) plus slack for cold caches. */
constexpr uint32_t kPrefetchStageRows = 192;
/* Each slot size class grows by its own chunks, so a smaller chunk bounds
 * the capacity a class can strand in a part-used chunk. */
constexpr uint64_t kChunkBytes = 128ull << 20;
constexpr uint64_t kReadBytes = 32ull << 20;
constexpr uint64_t kDirectAlign = 4096;
constexpr uint32_t kReadWorkers = 4;
constexpr uint32_t kRecentRows = 16;
/* Route rows of the largest split evaluation (decode, DSpark verification
 * and joint rows); larger batches read their experts in one pass. */
constexpr uint32_t kSplitRows = DS41_DECODER_MAX_SPECULATIVE * DS41_N_EXPERT_USED;
/* Batches of at least this many tokens adopt a wide (prefill-scale)
 * prediction job for their layer; smaller batches drop it instead of reading
 * a whole prompt's experts. */
constexpr uint32_t kSweepTokens = 1024;
constexpr uint32_t kSeedPerLayer = DS41_N_EXPERT;
constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();
constexpr uint64_t kTableEntries = uint64_t{DS41_N_LAYER} * DS41_N_EXPERT;
constexpr uint32_t kNoLayer = std::numeric_limits<uint32_t>::max();
constexpr std::array<uint16_t, DS41_N_LAYER * DS41_N_EXPERT> kDefaultExpertRank = {
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
  /* Atomic: the prefetch submission thread reads the target layer's slots
   * while the evaluation thread promotes or evicts entries. */
  std::atomic<uint32_t> slot{kNoSlot};
  uint64_t last = 0;
  double score = 0;
};

struct Slot {
  uint8_t* data = nullptr;
  uint32_t size_class = 0;  // capacity is the class's bytes
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
  /* Prefetch bookkeeping of the owning job; null for demand reads. */
  std::atomic<uint64_t>* job_bytes = nullptr;
  std::atomic<bool>* job_failed = nullptr;
};

/* A recorded router prediction: the ids live in decoder-owned pinned memory
 * and are valid until the event is waited on and the job submitted. */
struct Prediction {
  hipEvent_t event = nullptr;
  uint32_t layer = 0;
  const int32_t* ids = nullptr;
  uint32_t count = 0;
  bool wide = false;
};

void PredictWork(ds41_expert_store* store);

/* A background read job submitted between demand phases. Workers run its
 * tasks whenever no demand phase is active; the next prepare of the same
 * layer adopts the still-unrun tasks of the experts it routes into its
 * phase, any other prepare drops the job after the in-flight tasks drain.
 * Tasks are grouped per expert so adoption can skip unrouted experts:
 * waiting for a wrong prediction would cost the phase its whole read. */
struct PrefetchJob {
  std::vector<ReadTask> tasks;
  std::array<uint32_t, DS41_N_EXPERT> task_begin{};  // first task per expert
  std::array<uint32_t, DS41_N_EXPERT> task_count{};  // tasks per expert
  std::array<uint32_t, DS41_N_EXPERT> row_of{};  // compact stage row per expert
  std::vector<uint32_t> experts;  // experts the completed job stages
  std::atomic<uint64_t> bytes{0};
  std::atomic<bool> failed{false};
  uint32_t layer = 0;
  /* A prefill-scale prediction; only a batch-scale prepare adopts it, so a
   * decode step never inherits a whole-prompt expert set. */
  bool wide = false;
  size_t taken = 0;          // tasks already handed to workers
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
 * upload is awaited before its buffer is reused. Demand phases always
 * preempt the background prefetch sweep, which resumes between phases. No
 * work outlives Execute, so stage buffers and shard descriptors stay
 * valid. */
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

  /* Runs one demand phase for `layer`: resolves any background job (adopting
   * a same-layer, healthy job whose reads the batch will consume, parking a
   * next-layer job for after the phase, dropping anything overtaken), lets
   * `build` assemble the task list around the adopted job, then keeps the
   * workers on it while the caller's overlap callback runs. `adopt_wide`
   * gates prefill-scale jobs, which a small batch must drop instead of
   * reading a whole prompt's experts. */
  /* Resolves the background job before a demand phase of `layer` is
   * planned: quiesces its tasks and adopts a healthy same-layer job (a wide
   * one only when adopt_wide), parks a next-layer job to resume after the
   * phase, and drops anything overtaken. Callers need the adopted job's row
   * assignment to build the weight table, which is why this is separate
   * from RunPhase. A job submitted between Resolve and RunPhase runs as
   * background after the phase and is simply double-read if it targeted
   * this layer — harmless, and rarer than the prediction it loses. */
  std::shared_ptr<PrefetchJob> Resolve(uint32_t layer, bool adopt_wide,
                                       uint64_t* dropped_bytes) {
    std::unique_lock lock(mutex_);
    if (!pf_run_)
      return nullptr;
    std::shared_ptr<PrefetchJob> job;
    pf_stop_ = true;
    resolving_ = true;
    pf_idle_.wait(lock, [this] { return pf_inflight_ == 0; });
    pf_run_ = false;
    pf_stop_ = false;
    resolving_ = false;
    pf_->taken = pf_next_;
    if (pf_->layer == layer && !pf_->failed.load() && (adopt_wide || !pf_->wide))
      job = pf_;  // its un-run, routed tasks join the phase
    else if (pf_->layer == layer + 1 && !pf_->failed.load())
      parked_ = pf_;  // the next layer's parity half is free; resume after
    else
      *dropped_bytes += pf_->bytes.load();
    pf_.reset();
    pf_next_ = 0;
    return job;
  }

  bool RunPhase(uint32_t layer, const std::vector<ReadTask>& tasks,
                bool (*overlap)(void*, const ds41_expert_overlap*),
                void* overlap_context, const ds41_expert_overlap& info) {
    {
      std::lock_guard lock(mutex_);
      demand_ = &tasks;
      demand_next_ = 0;
      parked_workers_ = 0;
      phase_failed_.store(false);
      ++wake_;
      start_.notify_all();
    }
    const bool overlap_ok = !overlap || overlap(overlap_context, &info);
    {
      std::unique_lock lock(mutex_);
      done_.wait(lock, [this] { return parked_workers_ == threads_.size(); });
      demand_ = nullptr;
      parked_workers_ = 0;
      /* After the phase: a job submitted mid-phase runs next; otherwise the
       * parked next-layer job resumes. Both target layer + 1; anything else
       * was overtaken by events and is not queued in the first place. */
      std::shared_ptr<PrefetchJob> next;
      if (pending_ && pending_->layer == layer + 1)
        next = std::move(pending_);
      else if (parked_ && parked_->layer == layer + 1)
        next = std::move(parked_);
      pending_.reset();
      parked_.reset();
      if (next) {
        pf_ = std::move(next);
        pf_next_ = pf_->taken;  // a parked job resumes where it stopped
        pf_run_ = true;
        ++wake_;
      }
    }
    start_.notify_all();
    return overlap_ok && !phase_failed_.load();
  }

  bool SubmitPrefetch(std::shared_ptr<PrefetchJob> job) {
    std::unique_lock lock(mutex_);
    // A submission racing a phase's job resolution is simply dropped: the
    // prediction is best effort, and adopting half-resolved state is unsound.
    if (stopping_ || resolving_ || !job || job->tasks.empty())
      return false;
    if (demand_) {
      // Submitted mid-phase: run right after it, replacing any queued job
      // (a queued job has not run, so nothing is wasted).
      pending_ = std::move(job);
      return true;
    }
    if (pf_run_) {
      pf_stop_ = true;
      resolving_ = true;
      pf_idle_.wait(lock, [this] { return pf_inflight_ == 0; });
      pf_stop_ = false;
      resolving_ = false;
      orphaned_.fetch_add(pf_->bytes.load());
    }
    pf_ = std::move(job);
    pf_next_ = 0;
    pf_run_ = true;
    ++wake_;
    start_.notify_all();
    return true;
  }

  /* Bytes of jobs displaced before any phase consumed them; the store folds
   * them into its statistics from its own thread. */
  uint64_t TakeOrphaned() { return orphaned_.exchange(0); }

  /* Idle for the caller's fast path: no phase is running and no prefetch
   * task is queued or in flight. A job whose tasks all completed still
   * waits for the next phase to resolve it. */
  bool Idle() {
    std::lock_guard lock(mutex_);
    return !demand_ && !pending_ && pf_inflight_ == 0 &&
           (!pf_run_ || pf_next_ >= pf_->tasks.size());
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

  bool RunTask(const ReadTask& task, uint32_t index) {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(buffers_[index]);
    auto* buffer = reinterpret_cast<uint8_t*>((raw + kDirectAlign - 1) /
                                              kDirectAlign * kDirectAlign);
    uint64_t skip = 0;
    const bool direct =
        read_direct(task.direct_fd, buffer, task.bytes, task.offset, &skip);
    if (direct)
      ++direct_reads_;
    else {
      skip = 0;
      ++buffered_reads_;
    }
    if ((!direct && !read_all(task.fd, buffer, task.bytes, task.offset)) ||
        hipMemcpyAsync(task.destination, buffer + skip, task.bytes,
                       hipMemcpyHostToDevice, streams_[index]) != hipSuccess ||
        hipEventRecord(events_[index], streams_[index]) != hipSuccess ||
        hipEventSynchronize(events_[index]) != hipSuccess)
      return false;
    if (task.job_bytes)
      task.job_bytes->fetch_add(task.bytes);
    return true;
  }

  void Work(uint32_t index) {
    const bool device_ok = hipSetDevice(device_) == hipSuccess;
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock lock(mutex_);
      start_.wait(lock, [&] { return stopping_ || wake_ != seen; });
      if (stopping_)
        return;
      seen = wake_;
      lock.unlock();
      for (;;) {
        const ReadTask* task = nullptr;
        bool demand = false;
        {
          std::lock_guard lock(mutex_);
          if (stopping_)
            return;
          if (demand_) {
            if (demand_next_ < demand_->size()) {
              task = &(*demand_)[demand_next_++];
              demand = true;
            } else {
              /* Park once per phase: a worker that joined the phase from a
               * prefetch task still holds an older wake count and would
               * otherwise wake for this phase again and park twice, which
               * overshoots the count RunPhase waits for. */
              seen = wake_;
              if (++parked_workers_ == threads_.size())
                done_.notify_one();
              break;
            }
          } else if (pf_run_ && !pf_stop_ && pf_next_ < pf_->tasks.size()) {
            task = &pf_->tasks[pf_next_++];
            demand = false;
            ++pf_inflight_;
          } else {
            break;
          }
        }
        const bool ok = device_ok && RunTask(*task, index);
        std::lock_guard lock(mutex_);
        if (demand) {
          // Adopted prefetch tails carry job pointers but belong to the
          // phase, so their failure ends the evaluation like a demand read.
          if (!ok)
            phase_failed_.store(true);
        } else {
          --pf_inflight_;
          if (!ok && task->job_failed)
            task->job_failed->store(true);
          if (pf_stop_ && pf_inflight_ == 0)
            pf_idle_.notify_all();
        }
      }
    }
  }

  std::array<uint8_t*, kReadWorkers> buffers_;
  int device_;
  std::vector<std::thread> threads_;
  std::array<hipStream_t, kReadWorkers> streams_{};
  std::array<hipEvent_t, kReadWorkers> events_{};
  std::mutex mutex_;
  std::condition_variable start_, done_, pf_idle_;
  const std::vector<ReadTask>* demand_ = nullptr;
  size_t demand_next_ = 0;
  std::atomic<bool> phase_failed_{false};
  std::shared_ptr<PrefetchJob> pf_, pending_, parked_;
  bool pf_run_ = false, pf_stop_ = false, resolving_ = false;
  size_t pf_next_ = 0;
  uint32_t pf_inflight_ = 0;
  uint32_t parked_workers_ = 0;
  uint64_t wake_ = 0;
  bool stopping_ = false;
  std::atomic<uint64_t> direct_reads_{0}, buffered_reads_{0};
  std::atomic<uint64_t> orphaned_{0};
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
  /* The demand stage (whole-layer, rows indexed by expert id) and the
   * compact prefetch stage (two parity halves of job-relative rows).
   * Demand reads are ordered behind the previous layer's compute by the
   * blocking routes copy, so they safely share one buffer; background jobs
   * lack that ordering and write the compact stage instead, where the
   * parity halves keep a job from clobbering the rows the layer before
   * adopted. */
  std::array<std::array<uint8_t*, 3>, 2> stage{};
  /* Compact prefetch row stride per projection: the largest expert of any
   * layer. Rows must not use the layer's own size, or a layer with large
   * experts overruns into the other parity half when the layer after has
   * smaller ones (the builds mix 50- to 82-byte block types). */
  std::array<uint64_t, 3> stage_stride{};
  bool stage_pairs = false;
  /* The only layer a concurrent prefetch submission may target: the next
   * prepare call's. Set by that call before its reads begin. */
  std::atomic<uint32_t> pf_expected{kNoLayer};
  /* Router-predicted prefetch: the decoder records predictions with the
   * device event that signals their ids, and this thread submits each read
   * job when the event lands, so reads start during the layer the decoder
   * is still computing instead of when its phase would. */
  std::thread predictor;
  std::mutex pred_mutex;
  std::condition_variable pred_cv, pred_done;
  std::vector<Prediction> pred_queue;
  bool pred_active = false;
  bool pred_stopping = false;
  std::vector<void*> chunks;
  /* Per-layer device tables of each expert's weight base (slot data or stage
   * row) with pinned host mirrors of what was uploaded, and the hit/miss
   * route rows of the current prepare call (hit rows, then miss rows) with
   * their pinned staging. A layer's table is uploaded only when it changes,
   * which a warm cache almost never does; the uploads are asynchronous, as
   * the blocking routes copy of the next prepare orders them before any
   * mirror is rewritten. */
  std::array<const uint8_t**, 3> table_host{};
  std::array<const uint8_t**, 3> table{};
  int32_t* split = nullptr;
  int32_t* split_host = nullptr;
  std::array<int, DS41_MAX_SHARDS> direct_fds = [] {
    std::array<int, DS41_MAX_SHARDS> fds;
    fds.fill(-1);
    return fds;
  }();
  std::vector<Slot> slots;
  /* Slot sizes: the distinct per-layer expert sizes, ascending. The builds
   * mix block types, so one largest-expert slot size wasted 14-16% of the
   * cache. A slot holds any expert up to its class's size. */
  std::vector<uint64_t> class_bytes;
  std::array<uint32_t, DS41_N_LAYER> layer_class{};
  std::vector<std::vector<uint32_t>> free_slots;  // per class
  std::vector<uint64_t> grow_failure;             // tick, per class
  std::array<uint8_t*, kReadWorkers> read_buffers{};
  std::unique_ptr<ReadPool> readers;
  uint64_t cache_limit = 0;
  uint64_t tick = 0;
  uint64_t logged_tokens = 0;
  ds41_expert_store_stats stats;
  ds41_expert_store_stats logged_stats;

  ~ds41_expert_store() {
    {
      std::lock_guard lock(pred_mutex);
      pred_stopping = true;
    }
    pred_cv.notify_all();
    if (predictor.joinable())
      predictor.join();
    readers.reset();
    for (int fd : direct_fds)
      if (fd >= 0)
        (void)close(fd);
    (void)hipDeviceSynchronize();
    for (void* chunk : chunks)
      (void)hipFree(chunk);
    for (auto& set : stage)
      for (void* p : set)
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
    if (split)
      (void)hipFree(split);
    if (split_host)
      (void)hipHostFree(split_host);
  }

  bool grow(uint32_t size_class) {
    const uint64_t free_bytes = ds41_expert_available_memory();
    if (free_bytes <= kReserve)
      return false;
    uint64_t available = free_bytes - kReserve;
    if (stats.cache_bytes >= cache_limit)
      return false;
    available = std::min(available, cache_limit - stats.cache_bytes);
    const uint64_t slot_bytes = class_bytes[size_class];
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
      free_slots[size_class].push_back(static_cast<uint32_t>(slots.size()));
      slots.push_back({static_cast<uint8_t*>(allocation) + i * slot_bytes,
                       size_class, kNoSlot, kNoSlot});
    }
    return true;
  }

  uint32_t acquire(uint32_t layer, uint32_t expert, bool allow_evict) {
    const uint32_t own = layer_class[layer];
    if (free_slots[own].empty() && grow_failure[own] != tick && !grow(own))
      grow_failure[own] = tick;
    // Once the cache stops growing, the smallest larger free slot serves.
    for (uint32_t c = own; c < free_slots.size(); ++c)
      if (!free_slots[c].empty()) {
        const uint32_t slot = free_slots[c].back();
        free_slots[c].pop_back();
        return slot;
      }
    if (!allow_evict)
      return kNoSlot;
    uint32_t victim = kNoSlot;
    double lowest = std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < slots.size(); ++i) {
      const Slot& s = slots[i];
      if (s.layer == kNoSlot || s.size_class < own)
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

  void release(uint32_t slot_id) {
    Slot& slot = slots[slot_id];
    slot.layer = slot.expert = kNoSlot;
    free_slots[slot.size_class].push_back(slot_id);
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
    // Slot bases stay 256-byte aligned, as the chunk allocations are.
    combined = (combined + 255) / 256 * 256;
    if (std::find(store->class_bytes.begin(), store->class_bytes.end(),
                  combined) == store->class_bytes.end())
      store->class_bytes.push_back(combined);
  }
  std::sort(store->class_bytes.begin(), store->class_bytes.end());
  for (uint32_t layer = 0; layer < DS41_N_LAYER; ++layer) {
    uint64_t combined = 0;
    for (const Source& src : store->source[layer])
      combined += src.bytes;
    combined = (combined + 255) / 256 * 256;
    store->layer_class[layer] = static_cast<uint32_t>(
        std::lower_bound(store->class_bytes.begin(), store->class_bytes.end(),
                         combined) -
        store->class_bytes.begin());
  }
  store->free_slots.resize(store->class_bytes.size());
  store->grow_failure.assign(store->class_bytes.size(), ~0ull);
  bool ok = true;
  for (uint32_t i = 0; i < kReadWorkers && ok; ++i)
    ok = hipHostMalloc(&store->read_buffers[i],
                       kReadBytes + 2 * kDirectAlign) == hipSuccess;
  for (uint32_t p = 0; p < 3 && ok; ++p)
    ok =
        hipMalloc(&store->stage[0][p], max_bytes[p] * DS41_N_EXPERT) == hipSuccess;
  /* The second stage set (prefetch) is allocated lazily, after the startup
   * seed has taken its memory, so it never shrinks the resident set. */
  for (uint32_t p = 0; p < 3 && ok; ++p)
    ok = hipHostMalloc(reinterpret_cast<void**>(&store->table_host[p]),
                       kTableEntries * sizeof(const uint8_t*)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&store->table[p]),
                   kTableEntries * sizeof(const uint8_t*)) == hipSuccess;
  /* Null mirrors never match a weight base, so each layer's first prepare
   * uploads its table. */
  for (uint32_t p = 0; p < 3 && ok; ++p)
    std::fill_n(store->table_host[p], kTableEntries, nullptr);
  if (ok)
    ok = hipMalloc(reinterpret_cast<void**>(&store->split),
                   2 * kSplitRows * sizeof(int32_t)) == hipSuccess &&
         hipHostMalloc(reinterpret_cast<void**>(&store->split_host),
                       2 * kSplitRows * sizeof(int32_t)) == hipSuccess;
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
  try {
    store->predictor = std::thread(PredictWork, store);
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

bool seed_ranked(ds41_expert_store* store, std::vector<SeedItem> ranked,
                 const char* source) {
  /* Experts a histogram leaves out follow it, interleaved across layers, so
   * a cache larger than the traced set does not start part empty. The
   * built-in ranking orders all of them. */
  const size_t ranked_count = ranked.size();
  std::array<std::array<bool, DS41_N_EXPERT>, DS41_N_LAYER> ranked_set{};
  for (const SeedItem& item : ranked)
    ranked_set[item.layer][item.expert] = true;
  for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
    for (uint32_t l = 0; l < DS41_N_LAYER; ++l)
      if (!ranked_set[l][e])
        ranked.push_back({l, e, 0});
  const uint64_t available = ds41_expert_available_memory();
  const uint64_t free_budget = available > kReserve ? available - kReserve : 0;
  // Leave capacity for session allocations and for experts learned from the
  // actual request. At startup the slots are otherwise all free.
  const uint64_t budget = std::min(store->cache_limit, free_budget) / 16 * 15;
  size_t target = 0;
  for (uint64_t used = 0; target < ranked.size(); ++target) {
    used += store->class_bytes[store->layer_class[ranked[target].layer]];
    if (used > budget)
      break;
  }
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
        "expert seed (%s): cached %u of %zu experts (%zu ranked) from %.2f GiB "
        "in %.2f s",
        source, resident, target, std::min(target, ranked_count),
        static_cast<double>(bytes) / kGiB, seconds);
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
  return seed_ranked(store, std::move(ranked), "built-in V4.1 ranking");
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
  return seed_ranked(store, std::move(ranked), path);
}

namespace {

/* Demand read tasks into the whole-layer stage, merging consecutive experts
 * of one projection into chunks of at most kReadBytes and ordering chunks by
 * shard and offset for sequential access. */
std::vector<ReadTask> build_read_tasks(const ds41_expert_store* store, uint32_t layer,
                                       const std::array<bool, DS41_N_EXPERT>& wanted) {
  const auto& src = store->source[layer];
  std::array<uint32_t, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return src[a].fd == src[b].fd ? src[a].offset < src[b].offset
                                  : src[a].fd < src[b].fd;
  });
  std::vector<ReadTask> tasks;
  for (uint32_t p : order) {
    for (uint32_t e = 0; e < DS41_N_EXPERT;) {
      if (!wanted[e]) {
        ++e;
        continue;
      }
      const uint32_t first = e++;
      while (e < DS41_N_EXPERT && wanted[e])
        ++e;
      uint64_t offset = src[p].offset + (uint64_t)first * src[p].bytes;
      uint64_t left = (e - first) * src[p].bytes;
      uint8_t* dst = store->stage[0][p] + (uint64_t)first * src[p].bytes;
      while (left) {
        const uint64_t bytes = std::min(left, kReadBytes);
        tasks.push_back({src[p].fd, src[p].direct_fd, offset, bytes, dst});
        offset += bytes;
        dst += bytes;
        left -= bytes;
      }
    }
  }
  return tasks;
}

/* The compact prefetch stage is bought only when prefetch actually starts,
 * so it never shrinks the startup seed's memory budget. */
bool ensure_stage_pairs(ds41_expert_store* store) {
  if (store->stage_pairs)
    return true;
  std::array<uint64_t, 3> bytes{};
  for (uint32_t p = 0; p < 3; ++p)
    for (uint32_t layer = 0; layer < DS41_N_LAYER; ++layer)
      bytes[p] = std::max(bytes[p], store->source[layer][p].bytes);
  store->stage_stride = bytes;
  for (uint32_t p = 0; p < 3; ++p)
    if (hipMalloc(&store->stage[1][p],
                  bytes[p] * kPrefetchStageRows) != hipSuccess) {
      for (uint32_t q = 0; q <= p; ++q) {
        if (store->stage[1][q])
          (void)hipFree(store->stage[1][q]);
        store->stage[1][q] = nullptr;
      }
      return false;
    }
  store->stage_pairs = true;
  return true;
}

/* Prefetch tasks of one expert into a compact stage row: a chunked range
 * per projection, in the shard/offset-sorted projection order. */
void append_expert_tasks(const ds41_expert_store* store, uint32_t layer, uint32_t row,
                         uint32_t e, std::vector<ReadTask>& tasks) {
  const auto& src = store->source[layer];
  std::array<uint32_t, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return src[a].fd == src[b].fd ? src[a].offset < src[b].offset
                                  : src[a].fd < src[b].fd;
  });
  for (uint32_t p : order) {
    uint64_t offset = src[p].offset + (uint64_t)e * src[p].bytes;
    uint8_t* dst = store->stage[1][p] + (uint64_t)row * store->stage_stride[p];
    for (uint64_t left = src[p].bytes; left;) {
      const uint64_t bytes = std::min(left, kReadBytes);
      tasks.push_back({src[p].fd, src[p].direct_fd, offset, bytes, dst});
      offset += bytes;
      dst += bytes;
      left -= bytes;
    }
  }
}

std::shared_ptr<PrefetchJob> build_prefetch_job(
    const ds41_expert_store* store, uint32_t layer,
    const std::array<bool, DS41_N_EXPERT>& wanted, bool wide) {
  if (!store->stage_pairs)
    return nullptr;
  auto job = std::make_shared<PrefetchJob>();
  job->layer = layer;
  job->wide = wide;
  /* Tasks are grouped per expert at a compact row of the job's parity half,
   * so a later prepare can adopt exactly the routed experts' reads; experts
   * beyond the half simply stay demand reads. */
  constexpr uint32_t kHalf = kPrefetchStageRows / 2;
  const uint32_t base = (layer & 1) * kHalf;
  uint32_t used = 0;
  for (uint32_t e = 0; e < DS41_N_EXPERT && used < kHalf; ++e) {
    if (!wanted[e])
      continue;
    job->row_of[e] = base + used++;
    job->task_begin[e] = static_cast<uint32_t>(job->tasks.size());
    append_expert_tasks(store, layer, job->row_of[e], e, job->tasks);
    job->task_count[e] =
        static_cast<uint32_t>(job->tasks.size()) - job->task_begin[e];
    job->experts.push_back(e);
  }
  if (job->tasks.empty())
    return nullptr;
  for (ReadTask& task : job->tasks) {
    task.job_bytes = &job->bytes;
    task.job_failed = &job->failed;
  }
  return job;
}

void PredictWork(ds41_expert_store* store) {
  for (;;) {
    Prediction prediction{};
    {
      std::unique_lock lock(store->pred_mutex);
      store->pred_cv.wait(lock, [&] {
        return store->pred_stopping || !store->pred_queue.empty();
      });
      if (store->pred_stopping)
        return;
      prediction = store->pred_queue.front();
      store->pred_queue.erase(store->pred_queue.begin());
      store->pred_active = true;
    }
    if (hipEventSynchronize(prediction.event) == hipSuccess)
      (void)ds41_expert_store_prefetch(store, prediction.layer, prediction.ids,
                                       prediction.count, prediction.wide);
    {
      std::lock_guard lock(store->pred_mutex);
      store->pred_active = false;
      store->pred_done.notify_all();
    }
  }
}

}  // namespace

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
    /* The router drops a slot (-1) when its scores are not finite; the
     * kernels zero that row, as on the resident path. */
    if (id == -1)
      continue;
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
  /* Resolve the background job before planning the phase: the table and the
   * demand reads both need its compact row assignment. */
  uint32_t misses = 0;
  for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
    misses += missing[e] ? 1u : 0u;
  /* Predictions may target this layer or the next; after the last layer the
   * next evaluation's first prepare reopens the window. */
  store->pf_expected.store(layer + 1 < DS41_N_LAYER ? layer + 1 : 0,
                           std::memory_order_relaxed);
  const uint64_t orphaned = store->readers->TakeOrphaned();
  store->stats.bytes_read += orphaned;
  store->stats.prefetch_dropped_bytes += orphaned;
  std::shared_ptr<PrefetchJob> adopted;
  uint64_t dropped = 0;
  const bool phase = misses || !store->readers->Idle();
  if (phase)
    adopted = store->readers->Resolve(layer, n_tokens >= kSweepTokens, &dropped);
  store->stats.bytes_read += dropped;
  store->stats.prefetch_dropped_bytes += dropped;
  /* Map every expert of the layer to its weights: the cache slot when
   * resident, the adopted job's compact stage row when it stages one, else
   * the demand stage row the phase's reads fill. The kernels index the
   * table, so resident experts are read in place without a staging copy. */
  const auto& src = store->source[layer];
  std::array<const uint8_t**, 3> mirror{};
  std::array<const uint8_t**, 3> device_table{};
  for (uint32_t p = 0; p < 3; ++p) {
    mirror[p] = store->table_host[p] + uint64_t{layer} * DS41_N_EXPERT;
    device_table[p] = store->table[p] + uint64_t{layer} * DS41_N_EXPERT;
  }
  std::array<bool, 3> changed{};
  for (uint32_t e = 0; e < DS41_N_EXPERT; ++e) {
    const uint32_t slot_id = store->entries[layer][e].slot;
    const uint8_t* base =
        slot_id != kNoSlot ? store->slots[slot_id].data : nullptr;
    const bool covered = adopted && adopted->task_count[e] != 0;
    const uint32_t job_row = covered ? adopted->row_of[e] : 0;
    uint64_t staged = 0;
    for (uint32_t p = 0; p < 3; ++p) {
      const uint8_t* row =
          covered ? store->stage[1][p] + (uint64_t)job_row * store->stage_stride[p]
                  : store->stage[0][p] + (uint64_t)e * src[p].bytes;
      const uint8_t* entry = base ? base + staged : row;
      changed[p] = changed[p] || mirror[p][e] != entry;
      mirror[p][e] = entry;
      staged += src[p].bytes;
    }
  }
  for (uint32_t p = 0; p < 3; ++p)
    if (changed[p] &&
        hipMemcpyAsync(device_table[p], mirror[p],
                       DS41_N_EXPERT * sizeof(uint8_t*),
                       hipMemcpyHostToDevice, nullptr) != hipSuccess)
      return false;
  /* The hit and miss rows keep their route positions, each array holding -1
   * for the other's experts, so one launch over either array leaves exactly
   * the other launch's rows to fill. */
  ds41_expert_overlap info{};
  info.layer = layer;
  info.rows = static_cast<uint32_t>(n_routes);
  for (uint32_t p = 0; p < 3; ++p)
    info.table[p] = device_table[p];
  if (n_routes <= kSplitRows) {
    uint32_t miss_rows = 0;
    for (size_t i = 0; i < ids.size(); ++i)
      miss_rows += ids[i] >= 0 && missing[ids[i]];
    info.miss_rows = miss_rows;
    /* Only a batch with both hits and misses splits its launches; any other
     * batch keeps one pass over the routes, so its rows are not uploaded. */
    if (miss_rows && miss_rows < n_routes) {
      int32_t* hit = store->split_host;
      int32_t* miss = store->split_host + n_routes;
      for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        const bool is_miss = id >= 0 && missing[id];
        hit[i] = is_miss ? -1 : id;  // a dropped slot stays -1 in both
        miss[i] = is_miss ? id : -1;
      }
      info.hit_ids = store->split;
      info.miss_ids = store->split + n_routes;
      if (hipMemcpyAsync(store->split, store->split_host,
                         2 * n_routes * sizeof(int32_t), hipMemcpyHostToDevice,
                         nullptr) != hipSuccess)
        return false;
    }
  }
  if (phase) {
    // Demand reads of the misses no adopted job row covers; the job's
    // un-run tasks for the experts this batch routes join them.
    std::array<bool, DS41_N_EXPERT> want{};
    for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
      want[e] = missing[e] && !(adopted && adopted->task_count[e]);
    std::vector<ReadTask> tasks = build_read_tasks(store, layer, want);
    uint64_t demand_bytes = 0;
    for (const ReadTask& task : tasks)
      demand_bytes += task.bytes;
    if (adopted) {
      /* Adopt only the experts this batch routes: waiting for a wrong
       * prediction would cost the phase its whole read. Tasks the workers
       * already took (index < taken) are done; the rest run in-phase. */
      const uint32_t taken = static_cast<uint32_t>(adopted->taken);
      for (uint32_t e : adopted->experts) {
        if (!missing[e])
          continue;
        const uint32_t end = adopted->task_begin[e] + adopted->task_count[e];
        for (uint32_t i = std::max(adopted->task_begin[e], taken); i < end; ++i)
          tasks.push_back(adopted->tasks[i]);
      }
    }
    if (!store->readers->RunPhase(layer, tasks, overlap, overlap_context, info)) {
      fprintf(stderr, "ds41: failed to stage experts or run overlapped work\n");
      return false;
    }
    store->stats.bytes_read += demand_bytes;
    if (adopted) {
      const uint64_t bytes = adopted->bytes.load();
      store->stats.bytes_read += bytes;
      store->stats.prefetch_bytes += bytes;
    }
  } else if (overlap && !overlap(overlap_context, &info)) {
    return false;
  }
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
      if (hipMemcpyAsync(dst, mirror[p][e], src[p].bytes,
                         hipMemcpyDeviceToDevice, nullptr) != hipSuccess) {
        store->release(slot_id);
        return false;
      }
      dst += src[p].bytes;
    }
    slot.layer = layer;
    slot.expert = e;
    store->entries[layer][e].slot = slot_id;
  }
  *gate = device_table[0];
  *up = device_table[1];
  *down = device_table[2];
  store->stats.stall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
  if (layer == DS41_N_LAYER - 1 &&
      store->stats.tokens >= store->logged_tokens + 64) {
    const ds41_expert_store_stats s = store->stats;
    const ds41_expert_store_stats old = store->logged_stats;
    const uint64_t uses = s.uses - old.uses, hits = s.hits - old.hits;
    const uint64_t tokens = s.tokens - old.tokens;
    const uint64_t read = s.bytes_read - old.bytes_read;
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, "ds41",
        "expert cache interval: %.1f%% route hits, %.2f MiB read/token "
        "(%.1f%% prefetched), %.2f ms host wait/token, %.2f GiB allocated",
        uses ? 100.0 * static_cast<double>(hits) / uses : 0.0,
        tokens ? static_cast<double>(read) / tokens / (1u << 20) : 0.0,
        read ? 100.0 * static_cast<double>(s.prefetch_bytes - old.prefetch_bytes) /
                   read
             : 0.0,
        tokens ? static_cast<double>(s.stall_ns - old.stall_ns) / tokens / 1e6
               : 0.0,
        static_cast<double>(s.cache_bytes) / kGiB);
    store->logged_stats = s;
    store->logged_tokens = store->stats.tokens;
  }
  return true;
}

bool ds41_expert_store_prefetch(ds41_expert_store* store, uint32_t layer,
                                const int32_t* experts, uint32_t count, bool wide) {
  if (!store || layer >= DS41_N_LAYER || !experts || !count)
    return false;
  // Only the layer the next prepare handles, or its successor, is accepted:
  // an overtaken prediction would never be adopted, and an older layer's
  // rows may be in compute.
  const uint32_t expected = store->pf_expected.load(std::memory_order_relaxed);
  if (layer != expected && layer != expected + 1)
    return false;
  std::array<bool, DS41_N_EXPERT> want{};
  for (uint32_t i = 0; i < count; ++i) {
    const int32_t e = experts[i];
    if (e < 0 || e >= static_cast<int32_t>(DS41_N_EXPERT))
      return false;
    want[e] = store->entries[layer][e].slot == kNoSlot;
  }
  if (!ensure_stage_pairs(store))
    return false;
  std::shared_ptr<PrefetchJob> job = build_prefetch_job(store, layer, want, wide);
  if (!job)
    return false;
  return store->readers->SubmitPrefetch(std::move(job));
}

bool ds41_expert_store_predict(ds41_expert_store* store, uint32_t layer,
                               hipEvent_t done, const int32_t* ids, uint32_t count,
                               bool wide) {
  if (!store || !done || !ids || !count || layer >= DS41_N_LAYER)
    return false;
  std::lock_guard lock(store->pred_mutex);
  if (store->pred_stopping || store->pred_queue.size() >= 4)
    return false;
  for (const Prediction& pending : store->pred_queue)
    if (pending.layer == layer)
      return false;
  store->pred_queue.push_back({done, layer, ids, count, wide});
  store->pred_cv.notify_one();
  return true;
}

void ds41_expert_store_predict_flush(ds41_expert_store* store) {
  if (!store)
    return;
  std::unique_lock lock(store->pred_mutex);
  store->pred_done.wait(lock, [&] {
    return store->pred_queue.empty() && !store->pred_active;
  });
}

bool ds41_expert_store_prefetch_idle(const ds41_expert_store* store) {
  return store ? store->readers->Idle() : true;
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
      "host wait/token, %.2f GiB allocated, %.2f GiB prefetched (%.2f GiB "
      "dropped)",
      100.0 * static_cast<double>(s.hits) / s.uses,
      s.tokens ? static_cast<double>(s.bytes_read) / s.tokens / (1u << 20)
               : 0.0,
      s.tokens ? static_cast<double>(s.stall_ns) / s.tokens / 1e6 : 0.0,
      static_cast<double>(s.cache_bytes) / kGiB,
      static_cast<double>(s.prefetch_bytes) / kGiB,
      static_cast<double>(s.prefetch_dropped_bytes) / kGiB);
  gufo::server::Logger::LogFormatted(
      gufo::server::LogLevel::kInfo, "ds41",
      "expert reads: %llu direct, %llu buffered",
      static_cast<unsigned long long>(store->readers->direct_reads()),
      static_cast<unsigned long long>(store->readers->buffered_reads()));
}
