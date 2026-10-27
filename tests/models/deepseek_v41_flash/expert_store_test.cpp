// Exercise streaming from two synthetic shards with a deliberately small
// cache. Verify the bytes consumed by kernels, eviction and I/O failure.
#include "src/models/deepseek_v41_flash/runtime/expert_store.h"

#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/model_data_internal.h"

namespace {

constexpr std::array<uint64_t, 3> kSizes{512, 768, 1024};

/* Expert bytes of one projection; the mixed fixture halves odd layers, as
 * the real builds mix quantization types of different block sizes. */
uint64_t Size(bool mixed, uint32_t layer, uint32_t projection) {
  return mixed && layer % 2 ? kSizes[projection] / 2 : kSizes[projection];
}

uint8_t Byte(uint32_t layer, uint32_t projection, uint32_t expert,
             uint64_t index) {
  return static_cast<uint8_t>(layer * 13 + projection * 31 + expert * 7 +
                              index * 3);
}

struct Fixture {
  ds41_model model{};
  ds41_weights weights{};
  std::array<std::array<ds41_tensor, 3>, DS41_N_LAYER> tensors{};
  std::array<int, 2> fds{-1, -1};
  bool ok = true;

  explicit Fixture(bool mixed = false) {
    for (int& fd : fds) {
      char path[] = "/tmp/gufo-ds41-experts-XXXXXX";
      fd = mkstemp(path);
      if (fd < 0) {
        ok = false;
        return;
      }
      (void)unlink(path);
    }
    model.n_shards = 2;
    model.shard_fd[0] = fds[0];
    model.shard_fd[1] = fds[1];
    // A gap between virtual shard ranges, as in the real loader.
    model.shard_offset[1] = 1ull << 28;
    // Deliberately unaligned shard data makes the last expert end at an
    // unaligned EOF, exercising direct reads that cover an aligned superset.
    std::array<uint64_t, 2> offsets{64, 64};
    for (uint32_t layer = 0; layer < DS41_N_LAYER && ok; ++layer) {
      const uint32_t shard = layer >= DS41_N_LAYER / 2 ? 1 : 0;
      auto& w = weights.layer[layer];
      w.ffn_gate_exps = &tensors[layer][0];
      w.ffn_up_exps = &tensors[layer][1];
      w.ffn_down_exps = &tensors[layer][2];
      for (uint32_t p = 0; p < 3 && ok; ++p) {
        auto& tensor = tensors[layer][p];
        tensor.abs_offset = model.shard_offset[shard] + offsets[shard];
        const uint64_t size = Size(mixed, layer, p);
        tensor.bytes = size * DS41_N_EXPERT;
        std::vector<uint8_t> bytes(tensor.bytes);
        for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
          for (uint64_t i = 0; i < size; ++i)
            bytes[e * size + i] = Byte(layer, p, e, i);
        ok = pwrite(fds[shard], bytes.data(), bytes.size(),
                    static_cast<off_t>(offsets[shard])) ==
             static_cast<ssize_t>(bytes.size());
        offsets[shard] += tensor.bytes;
      }
    }
    model.shard_size[0] = offsets[0];
    model.shard_size[1] = offsets[1];
  }

  ~Fixture() {
    for (int fd : fds)
      if (fd >= 0)
        (void)close(fd);
  }
};

bool CheckStage(uint32_t layer, const std::vector<int32_t>& ids,
                const std::array<const void*, 3>& table, bool mixed = false) {
  for (uint32_t p = 0; p < 3; ++p) {
    const auto* const* entries = static_cast<const uint8_t* const*>(table[p]);
    std::vector<const uint8_t*> rows(DS41_N_EXPERT);
    if (hipMemcpy(rows.data(), entries, rows.size() * sizeof(rows[0]),
                  hipMemcpyDeviceToHost) != hipSuccess)
      return false;
    std::vector<uint8_t> bytes(Size(mixed, layer, p));
    for (int32_t e : ids) {
      if (hipMemcpy(bytes.data(), rows[e], bytes.size(),
                    hipMemcpyDeviceToHost) != hipSuccess)
        return false;
      for (uint64_t i = 0; i < bytes.size(); ++i)
        if (bytes[i] != Byte(layer, p, static_cast<uint32_t>(e), i))
          return false;
    }
  }
  return true;
}

/* Copy of the device hit/miss rows of the last overlap. */
bool ReadRows(const int32_t* device, uint32_t rows, std::vector<int32_t>& out) {
  out.resize(rows);
  return hipMemcpy(out.data(), device, rows * sizeof(int32_t),
                   hipMemcpyDeviceToHost) == hipSuccess;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
    return 77;
  Fixture f;
  if (!f.ok)
    return 1;
  constexpr uint64_t kSlot = 512 + 768 + 1024;
  ds41_expert_store* store =
      ds41_expert_store_create(&f.model, &f.weights, 4 * kSlot);
  int32_t* routes = nullptr;
  if (!store || hipMalloc(&routes, 1024 * 6 * sizeof(int32_t)) != hipSuccess)
    return 1;
  std::array<const void*, 3> stage{};
  const auto run = [&](uint32_t layer, const std::vector<int32_t>& ids) {
    if (ids.empty() || ids.size() % 6 || ids.size() > 1024 * 6 ||
        hipMemcpy(routes, ids.data(), ids.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice) != hipSuccess)
      return false;
    return ds41_expert_store_prepare(store, layer, routes,
                                     static_cast<uint32_t>(ids.size() / 6),
                                     &stage[0], &stage[1], &stage[2]) &&
           CheckStage(layer, ids, stage);
  };
  bool ok = true;
  const auto check = [&](bool pass, const char* label) {
    if (!pass)
      std::fprintf(stderr, "expert store check failed: %s\n", label);
    ok = pass && ok;
  };
  check(run(0, {0, 1, 2, 3, 1, 2}), "cold stage bytes");
  const auto cold = ds41_expert_store_get_stats(store);
  check(cold.uses == 6 && cold.hits == 0 && cold.bytes_read == 4 * kSlot,
        "cold stats");
  check(run(0, {2, 3, 0, 1, 2, 1}), "warm stage bytes");
  const auto warm = ds41_expert_store_get_stats(store);
  check(warm.hits == 6 && warm.bytes_read == cold.bytes_read, "warm stats");
  int overlaps = 0;
  const auto overlap = [](void* context, const ds41_expert_overlap*) {
    ++*static_cast<int*>(context);
    return true;
  };
  const std::vector<int32_t> hot{0, 1, 2, 3, 1, 2};
  (void)hipMemcpy(routes, hot.data(), hot.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice);
  check(ds41_expert_store_prepare(store, 0, routes, 1, &stage[0], &stage[1],
                                  &stage[2], overlap, &overlaps) &&
            overlaps == 1 && CheckStage(0, hot, stage),
        "overlap with cache hits");
  const auto fail_overlap = [](void* context, const ds41_expert_overlap*) {
    ++*static_cast<int*>(context);
    return false;
  };
  const std::vector<int32_t> cold_overlap{7, 7, 7, 7, 7, 7};
  (void)hipMemcpy(routes, cold_overlap.data(),
                  cold_overlap.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  check(!ds41_expert_store_prepare(store, 1, routes, 1, &stage[0], &stage[1],
                                   &stage[2], fail_overlap, &overlaps) &&
            overlaps == 2 && run(1, cold_overlap),
        "failed overlap drains reads");
  // Repeated use of a new expert must eventually evict a colder entry, and
  // its three matrices must remain correct when read from the second shard.
  for (int i = 0; i < 3; ++i)
    check(run(25, {9, 9, 9, 9, 9, 9}), "eviction stage bytes");
  const auto evicted = ds41_expert_store_get_stats(store);
  check(evicted.hits > warm.hits && evicted.cache_bytes == 4 * kSlot,
        "eviction stats");
  // A mixed batch reports the hit and miss rows of a split launch.
  {
    ds41_expert_overlap info{};
    const auto capture = [](void* context, const ds41_expert_overlap* seen) {
      if (seen)
        *static_cast<ds41_expert_overlap*>(context) = *seen;
      return true;
    };
    const std::vector<int32_t> mixed{9, 9, 9, 20, 20, 21};
    (void)hipMemcpy(routes, mixed.data(), mixed.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice);
    std::array<const void*, 3> split_stage{};
    check(ds41_expert_store_prepare(store, 25, routes, 1, &split_stage[0],
                                    &split_stage[1], &split_stage[2], capture,
                                    &info) &&
              CheckStage(25, mixed, split_stage),
          "split stage bytes");
    std::vector<int32_t> hit, miss;
    check(info.miss_rows == 3 && info.rows == 6 && info.hit_ids &&
              info.miss_ids && ReadRows(info.hit_ids, 6, hit) &&
              ReadRows(info.miss_ids, 6, miss) &&
              hit == std::vector<int32_t>{9, 9, 9, -1, -1, -1} &&
              miss == std::vector<int32_t>{-1, -1, -1, 20, 20, 21},
          "split rows");
  }
  const std::vector<int32_t> invalid{0, 1, 2, 3, 4, 384};
  (void)hipMemcpy(routes, invalid.data(), invalid.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice);
  check(!ds41_expert_store_prepare(store, 0, routes, 1, &stage[0], &stage[1],
                                   &stage[2]),
        "invalid route");
  // A prefill scan fills empty slots and promotes its final rows. Once full,
  // another scan does not displace the working set for one-use experts.
  ds41_expert_store_destroy(store);
  store = ds41_expert_store_create(&f.model, &f.weights, 4 * kSlot);
  std::vector<int32_t> scan(32 * 6);
  for (int i = 0; i < 16 * 6; ++i)
    scan[i] = i;
  for (int i = 16 * 6; i < 32 * 6; ++i)
    scan[i] = 200;
  check(store && run(5, scan), "first scan stage bytes");
  auto before = ds41_expert_store_get_stats(store);
  check(run(5, {200, 200, 200, 200, 200, 200}), "promoted stage bytes");
  auto after = ds41_expert_store_get_stats(store);
  check(after.hits - before.hits == 6, "last-row promotion");
  for (int i = 0; i < 16 * 6; ++i)
    scan[i] = 100 + i;
  check(run(6, scan), "second scan stage bytes");
  before = ds41_expert_store_get_stats(store);
  check(run(6, {100, 100, 100, 100, 100, 100}), "one-use stage bytes");
  after = ds41_expert_store_get_stats(store);
  check(after.hits == before.hits, "scan-resistant admission");
  ds41_expert_store_destroy(store);
  // Background prefetch: the listed experts stage without a prepare, the
  // next prepare of that layer adopts them without re-reading, and any other
  // layer drops the job and reads its own experts again.
  store = ds41_expert_store_create(&f.model, &f.weights, 4 * kSlot);
  check(store && run(0, {0, 1, 2, 3, 1, 2}), "prefetch setup stage bytes");
  const auto sweep_base = ds41_expert_store_get_stats(store);
  const auto idle = [](const ds41_expert_store* s) {
    for (int i = 0; i < 2000 && !ds41_expert_store_prefetch_idle(s); ++i)
      usleep(1000);
    return ds41_expert_store_prefetch_idle(s);
  };
  const int32_t none[]{7};
  const int32_t first[]{5, 9};
  const int32_t second[]{7, 8};
  check(ds41_expert_store_prefetch(store, 3, none, 1) == false,
        "prefetch before the layer is expected");
  check(ds41_expert_store_prefetch(store, 1, first, 2) && idle(store),
        "prefetch submits and drains");
  auto stats = ds41_expert_store_get_stats(store);
  check(stats.bytes_read == sweep_base.bytes_read,
        "drained prefetch not folded yet");
  check(run(1, {5, 9, 5, 9, 5, 9}), "adopted prefetch stage bytes");
  stats = ds41_expert_store_get_stats(store);
  check(stats.bytes_read - sweep_base.bytes_read == 2 * kSlot &&
            stats.prefetch_bytes - sweep_base.prefetch_bytes == 2 * kSlot,
        "adoption reads each expert once");
  check(run(2, {30, 30, 30, 30, 30, 30}), "unrelated layer stage bytes");
  stats = ds41_expert_store_get_stats(store);
  const uint64_t after_two = stats.bytes_read;
  check(ds41_expert_store_prefetch(store, 3, second, 2) && idle(store),
        "second prefetch submits and drains");
  check(run(4, {11, 11, 11, 11, 11, 11}), "overtaking layer stage bytes");
  stats = ds41_expert_store_get_stats(store);
  check(stats.prefetch_dropped_bytes == 2 * kSlot &&
            stats.bytes_read - after_two == 2 * kSlot + kSlot,
        "overtaken prefetch dropped and reread");
  check(run(3, {7, 7, 7, 7, 7, 7}), "dropped prefetch expert stage bytes");
  stats = ds41_expert_store_get_stats(store);
  check(stats.prefetch_dropped_bytes == 2 * kSlot &&
            stats.prefetch_bytes == 2 * kSlot,
        "dropped bytes never counted as consumed");
  ds41_expert_store_destroy(store);
  // A prefill-scale (wide) prefetch stages a large expert set that a
  // batch-scale prepare adopts without rereading; a small batch drops it.
  store = ds41_expert_store_create(&f.model, &f.weights, 4 * kSlot);
  {
    std::vector<int32_t> all(DS41_N_EXPERT);
    for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
      all[e] = static_cast<int32_t>(e);
    std::vector<int32_t> scan(1024 * 6);
    for (uint32_t i = 0; i < scan.size(); ++i)
      scan[i] = static_cast<int32_t>(i % 8);
    check(store && run(0, scan), "wide source layer stage bytes");
    check(
        ds41_expert_store_prefetch(store, 1, all.data(), DS41_N_EXPERT, true) &&
            idle(store),
        "wide prefetch submits and drains");
    stats = ds41_expert_store_get_stats(store);
    const uint64_t demand_only = stats.bytes_read;
    constexpr uint32_t kHalf = 96;  // one parity half of the compact stage
    check(run(1, scan), "wide adoption stage bytes");
    stats = ds41_expert_store_get_stats(store);
    check(stats.bytes_read - demand_only == kHalf * kSlot &&
              stats.prefetch_bytes == kHalf * kSlot,
          "batch-scale prepare adopts the truncated wide prefetch");
    check(run(2, scan), "wide drop source layer stage bytes");
    check(
        ds41_expert_store_prefetch(store, 3, all.data(), DS41_N_EXPERT, true) &&
            idle(store),
        "second wide prefetch submits and drains");
    stats = ds41_expert_store_get_stats(store);
    const uint64_t reswept = stats.bytes_read;
    const std::vector<int32_t> one{40, 40, 40, 40, 40, 40};
    check(run(3, one), "small batch after wide stage bytes");
    stats = ds41_expert_store_get_stats(store);
    check(stats.bytes_read == reswept + kSlot + kHalf * kSlot &&
              stats.prefetch_dropped_bytes == kHalf * kSlot,
          "small batch drops the wide prefetch");
  }
  ds41_expert_store_destroy(store);
  store = ds41_expert_store_create(&f.model, &f.weights, 4 * kSlot);
  char histogram_path[] = "/tmp/gufo-ds41-histogram-XXXXXX";
  const int histogram = mkstemp(histogram_path);
  const char data[] =
      "DS41_EXPERT_HISTOGRAM_V1\n4 12 50\n4 13 10\n5 14 40\n6 15 1\n";
  check(histogram >= 0 &&
            write(histogram, data, sizeof(data) - 1) == sizeof(data) - 1,
        "write histogram");
  if (histogram >= 0)
    (void)close(histogram);
  check(store && ds41_expert_store_seed(store, histogram_path),
        "seed histogram");
  (void)unlink(histogram_path);
  before = ds41_expert_store_get_stats(store);
  check(before.uses == 0 && before.bytes_read == 0,
        "seed excluded from inference statistics");
  check(run(4, {12, 13, 12, 13, 12, 13}), "seeded stage bytes");
  after = ds41_expert_store_get_stats(store);
  check(after.hits == 6 && after.bytes_read == 0, "seeded cache hits");
  before = after;
  check(run(5, {14, 14, 14, 14, 14, 14}), "global ranked seed bytes");
  after = ds41_expert_store_get_stats(store);
  check(after.hits - before.hits == 6, "higher ranked layer seeded first");
  before = after;
  check(run(6, {15, 15, 15, 15, 15, 15}), "unseeded expert bytes");
  after = ds41_expert_store_get_stats(store);
  check(after.hits == before.hits, "seed leaves a slot for request experts");
  ds41_expert_store_destroy(store);
  store = ds41_expert_store_create(&f.model, &f.weights, 4 * kSlot);
  check(store && ds41_expert_store_seed_default(store),
        "seed built-in ranking");
  before = ds41_expert_store_get_stats(store);
  check(run(26, {116, 116, 116, 116, 116, 116}), "built-in seeded stage bytes");
  after = ds41_expert_store_get_stats(store);
  check(after.hits - before.hits == 6 && after.bytes_read == before.bytes_read,
        "built-in ranked expert is resident");
  check(run(39, {383, 383, 383, 383, 383, 383}), "unaligned EOF stage bytes");
  ds41_expert_store_log(store);
  // A short read must fail the evaluation, not admit uninitialized bytes.
  check(ftruncate(f.fds[1], 0) == 0, "truncate fixture");
  const std::vector<int32_t> absent{20, 21, 22, 23, 24, 25};
  (void)hipMemcpy(routes, absent.data(), absent.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice);
  check(!ds41_expert_store_prepare(store, 30, routes, 1, &stage[0], &stage[1],
                                   &stage[2]),
        "short read");
  ds41_expert_store_destroy(store);
  // Layers with different expert sizes share the compact prefetch stage. A
  // wide job for the next layer must leave the rows this layer adopted
  // intact, so rows are strided by the largest expert of any layer.
  {
    Fixture m(true);
    check(m.ok, "mixed-size fixture");
    store = ds41_expert_store_create(&m.model, &m.weights, 4 * kSlot);
    std::vector<int32_t> all(DS41_N_EXPERT);
    for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
      all[e] = static_cast<int32_t>(e);
    std::vector<int32_t> adopted(1024 * 6);
    for (uint32_t i = 0; i < adopted.size(); ++i)
      adopted[i] = static_cast<int32_t>(i % 96);  // one parity half
    const auto prepare = [&](uint32_t layer, const std::vector<int32_t>& ids) {
      return hipMemcpy(routes, ids.data(), ids.size() * sizeof(int32_t),
                       hipMemcpyHostToDevice) == hipSuccess &&
             ds41_expert_store_prepare(store, layer, routes,
                                       static_cast<uint32_t>(ids.size() / 6),
                                       &stage[0], &stage[1], &stage[2]);
    };
    check(store && prepare(1, adopted), "mixed source layer");
    check(
        ds41_expert_store_prefetch(store, 2, all.data(), DS41_N_EXPERT, true) &&
            idle(store),
        "mixed wide prefetch of the larger layer");
    check(prepare(2, adopted) && CheckStage(2, adopted, stage, true),
          "mixed wide adoption stage bytes");
    check(
        ds41_expert_store_prefetch(store, 3, all.data(), DS41_N_EXPERT, true) &&
            idle(store),
        "mixed wide prefetch of the smaller next layer");
    check(CheckStage(2, adopted, stage, true),
          "next layer's prefetch keeps the adopted rows");
    ds41_expert_store_destroy(store);
  }
  (void)hipFree(routes);
  std::printf("expert store: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
