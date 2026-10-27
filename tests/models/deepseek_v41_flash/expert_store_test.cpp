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

  Fixture() {
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
        tensor.bytes = kSizes[p] * DS41_N_EXPERT;
        std::vector<uint8_t> bytes(tensor.bytes);
        for (uint32_t e = 0; e < DS41_N_EXPERT; ++e)
          for (uint64_t i = 0; i < kSizes[p]; ++i)
            bytes[e * kSizes[p] + i] = Byte(layer, p, e, i);
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
                const std::array<const void*, 3>& table) {
  for (uint32_t p = 0; p < 3; ++p) {
    const auto* const* entries = static_cast<const uint8_t* const*>(table[p]);
    std::vector<uint8_t> bytes(kSizes[p]);
    for (int32_t e : ids) {
      if (hipMemcpy(bytes.data(), entries[e], bytes.size(),
                    hipMemcpyDeviceToHost) != hipSuccess)
        return false;
      for (uint64_t i = 0; i < kSizes[p]; ++i)
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
  if (!store || hipMalloc(&routes, 32 * 6 * sizeof(int32_t)) != hipSuccess)
    return 1;
  std::array<const void*, 3> stage{};
  const auto run = [&](uint32_t layer, const std::vector<int32_t>& ids) {
    if (ids.empty() || ids.size() % 6 || ids.size() > 32 * 6 ||
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
  (void)hipFree(routes);
  std::printf("expert store: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
