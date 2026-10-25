/* Opens a DeepSeek V4.1 Flash build and reports its memory plan. With
 * --upload the resident weights are copied to the GPU and the memory peak is
 * reported; otherwise the artifacts are only parsed, validated and bound. */
#include <glob.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/model.h"
#include "src/models/deepseek_v41_flash/runtime/v41_dspark.h"

namespace {
std::uint64_t MeminfoKiB(const char* key) {
  std::ifstream in("/proc/meminfo");
  std::string name;
  std::uint64_t value = 0;
  std::string unit;
  while (in >> name >> value >> unit)
    if (name == key)
      return value;
  return 0;
}

std::uint64_t StatusKiB(const char* key) {
  std::ifstream in("/proc/self/status");
  std::string line;
  const std::size_t len = std::strlen(key);
  while (std::getline(in, line))
    if (line.compare(0, len, key) == 0)
      return std::stoull(line.substr(len + 1));
  return 0;
}

std::uint64_t GttUsedBytes() {
  glob_t found{};
  std::uint64_t bytes = 0;
  if (::glob("/sys/class/drm/card*/device/mem_info_gtt_used", 0, nullptr,
             &found) == 0) {
    for (std::size_t i = 0; i < found.gl_pathc; ++i) {
      std::ifstream in(found.gl_pathv[i]);
      std::uint64_t value = 0;
      if (in >> value)
        bytes += value;
    }
  }
  ::globfree(&found);
  return bytes;
}

struct Dump {
  std::string dir;
  uint32_t pos = 0;
  FILE* routes = nullptr;
};

void DumpLayer(void* ctx, uint32_t layer, const float* hidden,
               const int32_t* experts) {
  auto* dump = static_cast<Dump*>(ctx);
  const std::string path = dump->dir + "/pos" + std::to_string(dump->pos) +
                           "_layer" + std::to_string(layer) + ".f32";
  if (FILE* f = std::fopen(path.c_str(), "wb")) {
    std::fwrite(hidden, sizeof(float), 4u * 5120u, f);
    std::fclose(f);
  }
  std::fprintf(dump->routes, "%u %u %d %d %d %d %d %d\n", dump->pos, layer,
               experts[0], experts[1], experts[2], experts[3], experts[4],
               experts[5]);
}

void Report(const char* stage) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  std::printf(
      "%-10s MemAvailable %.2f GiB, GTT used %.2f GiB, RSS %.2f GiB (peak %.2f "
      "GiB)\n",
      stage, MeminfoKiB("MemAvailable:") * 1024.0 / kGiB, GttUsedBytes() / kGiB,
      StatusKiB("VmRSS:") * 1024.0 / kGiB, StatusKiB("VmHWM:") * 1024.0 / kGiB);
}
int Argmax(const float* row, std::size_t vocab) {
  int best = 0;
  for (std::size_t i = 1; i < vocab; ++i)
    if (row[i] > row[(std::size_t)best])
      best = (int)i;
  return best;
}

/* Prefills `prompt` into a new decoder in batches. */
ds41_decoder* Prefill(ds41_engine* engine, const std::vector<int>& prompt,
                      uint32_t context, std::vector<float>* last) {
  ds41_decoder* d = ds41_decoder_create(engine, context);
  if (!d)
    return nullptr;
  const uint32_t cap = ds41_decoder_batch_capacity(d);
  for (uint32_t at = 0; at < prompt.size();) {
    const uint32_t n = std::min<uint32_t>(cap, (uint32_t)prompt.size() - at);
    const bool final = at + n == prompt.size();
    if (ds41_decoder_eval(d, prompt.data() + at, n, last->data(), final ? 1 : 0,
                          nullptr, nullptr) != 0) {
      ds41_decoder_destroy(d);
      return nullptr;
    }
    at += n;
  }
  return d;
}

/* --spec-check: speculative batches of four rows keeping 1, 2, 3, 4 rows in
 * turn. Pass A drafts the greedy continuation, pass B replaces the rows past
 * the kept prefix with other tokens. Exact rollback leaves both passes with
 * bit-identical logits for every kept row.
 * --spec-bench N: times speculative batches of N rows (0: 1 to 8), commit 1. */
int SpeculativeProbe(ds41_engine* engine, const std::vector<int>& prompt,
                     uint32_t context, int cycles, int bench) {
  const std::size_t vocab = (std::size_t)ds41_engine_vocab_size(engine);
  context = std::max<uint32_t>(context, (uint32_t)prompt.size() + 4096);
  std::vector<float> last(vocab);
  if (bench >= 0) {
    ds41_decoder* d = Prefill(engine, prompt, context, &last);
    if (!d)
      return 1;
    std::vector<float> rows(vocab * DS41_DECODER_MAX_SPECULATIVE);
    int token = Argmax(last.data(), vocab);
    for (uint32_t n = bench ? (uint32_t)bench : 1;
         n <= (bench ? (uint32_t)bench : 8); ++n) {
      std::vector<double> ms;
      for (int rep = 0; rep < 7; ++rep) {
        /* The greedy token, then prompt text: distinct tokens route to as
         * many experts as a real draft. */
        std::vector<int> batch(n, token);
        for (uint32_t i = 1; i < n; ++i)
          batch[i] = prompt[(97 * (std::size_t)rep + i) % prompt.size()];
        const auto t0 = std::chrono::steady_clock::now();
        if (ds41_decoder_eval_speculative(d, batch.data(), n, rows.data()) ||
            ds41_decoder_commit(d, 1))
          return 1;
        ms.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
        token = Argmax(rows.data(), vocab);
        /* DS41_PROBE_GAP=1 separates repetitions by an idle gap for GPU
         * profiles; it lowers the clocks, so timings run without it. */
        if (std::getenv("DS41_PROBE_GAP"))
          std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }
      std::sort(ms.begin() + 1, ms.end());  // the first repetition warms up
      std::printf("spec-bench n=%u at %u: median %.1f ms min %.1f ms\n", n,
                  ds41_decoder_pos(d), ms[1 + ms.size() / 2 - 1], ms[1]);
    }
    ds41_decoder_destroy(d);
    return 0;
  }
  /* Greedy continuation for pass A's drafts, from single-token steps. */
  std::vector<int> greedy;
  std::vector<std::vector<float>> greedy_logits;
  {
    ds41_decoder* d = Prefill(engine, prompt, context, &last);
    if (!d)
      return 1;
    greedy.push_back(Argmax(last.data(), vocab));
    for (int i = 0; i < cycles * 4 + 4; ++i) {
      if (ds41_decoder_step(d, greedy.back(), last.data(), nullptr, nullptr))
        return 1;
      greedy_logits.push_back(last);
      greedy.push_back(Argmax(last.data(), vocab));
    }
    ds41_decoder_destroy(d);
  }
  std::vector<std::vector<float>> kept[2];
  std::vector<int> kept_tokens[2];
  for (int pass = 0; pass < 2; ++pass) {
    ds41_decoder* d = Prefill(engine, prompt, context, &last);
    if (!d)
      return 1;
    std::vector<float> rows(vocab * 4);
    std::size_t at = 0;  // greedy index of the next row's token
    for (int c = 0; c < cycles; ++c) {
      const uint32_t keep = 1 + (uint32_t)c % 4;
      int batch[4];
      for (uint32_t r = 0; r < 4; ++r)
        batch[r] =
            pass == 1 && r >= keep ? (int)(1000 + 37 * r + c) : greedy[at + r];
      if (ds41_decoder_eval_speculative(d, batch, 4, rows.data()) ||
          ds41_decoder_commit(d, keep))
        return 1;
      for (uint32_t r = 0; r < keep; ++r) {
        kept[pass].emplace_back(
            rows.begin() + (std::ptrdiff_t)(r * vocab),
            rows.begin() + (std::ptrdiff_t)((r + 1) * vocab));
        kept_tokens[pass].push_back(Argmax(rows.data() + r * vocab, vocab));
      }
      at += keep;
    }
    ds41_decoder_destroy(d);
  }
  std::size_t identical = 0, agree = 0, as_single = 0;
  double worst = 0.0, worst_single = 0.0;
  for (std::size_t i = 0; i < kept[0].size(); ++i) {
    double diff = 0.0, diff_single = 0.0;
    for (std::size_t v = 0; v < vocab; ++v) {
      diff = std::max(diff, (double)std::fabs(kept[0][i][v] - kept[1][i][v]));
      diff_single = std::max(
          diff_single, (double)std::fabs(kept[0][i][v] - greedy_logits[i][v]));
    }
    identical += diff == 0.0;
    as_single += diff_single == 0.0;
    worst = std::max(worst, diff);
    worst_single = std::max(worst_single, diff_single);
    /* Row i of pass A predicts greedy[i + 1] under exact single-token decode.
     */
    agree += kept_tokens[0][i] == greedy[i + 1];
  }
  std::printf(
      "spec-check: %zu kept rows, %zu bit-identical between passes (max |diff| "
      "%.3g); against single-token decode: argmax matches on %zu, logits "
      "bit-identical on %zu (max |diff| %.3g)\n",
      kept[0].size(), identical, worst, agree, as_single, worst_single);
  return identical == kept[0].size() ? 0 : 1;
}
}  // namespace

/* --dspark-generate N: greedy decode of N tokens after the prompt, first
 * token by token, then in DSpark cycles; the outputs must match.
 * --dspark-sweep repeats the DSpark run with at most 1 to 5 drafts. */
int DsparkProbe(ds41_engine* engine, const std::vector<int>& prompt,
                uint32_t context, int n_tokens, bool sweep) {
  const int ctx =
      (int)std::max<std::size_t>(context, prompt.size() + n_tokens + 16);
  ds41_session* session = nullptr;
  if (ds41_session_create(&session, engine, ctx, true) != 0) {
    std::fprintf(stderr, "DSpark session creation failed\n");
    return 1;
  }
  char error[256] = {};
  ds41_tokens tokens{const_cast<int*>(prompt.data()), (int)prompt.size(),
                     (int)prompt.size()};
  const auto now = [] { return std::chrono::steady_clock::now(); };
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  if (ds41_session_sync(session, &tokens, error, sizeof(error)) != 0) {
    std::fprintf(stderr, "sync failed: %s\n", error);
    return 1;
  }
  std::vector<int> plain;
  auto t0 = now();
  while ((int)plain.size() < n_tokens) {
    const int token = ds41_session_argmax(session);
    plain.push_back(token);
    if (ds41_session_eval(session, token, error, sizeof(error)) != 0) {
      std::fprintf(stderr, "eval failed: %s\n", error);
      return 1;
    }
  }
  const double plain_ms = ms(t0, now());
  ds41_session_invalidate(session);
  if (ds41_session_sync(session, &tokens, error, sizeof(error)) != 0) {
    std::fprintf(stderr, "sync failed: %s\n", error);
    return 1;
  }
  const int widths = sweep ? 5 : 1;
  for (int width = 1; width <= widths; width++) {
    if (width > 1) {
      ds41_session_invalidate(session);
      if (ds41_session_sync(session, &tokens, error, sizeof(error)) != 0)
        return 1;
    }
    std::vector<int> spec;
    int block[32];
    int produced = 0;
    uint64_t base_drafted = 0, base_accepted = 0, base_steps = 0, base_full = 0;
    ds41_session_dspark_stats(session, nullptr, nullptr, &base_drafted,
                              &base_accepted, nullptr, nullptr, &base_full,
                              &base_steps, nullptr, nullptr);
    t0 = now();
    while ((int)spec.size() < n_tokens) {
      ds41_session_dspark_batch_item item{
          session,
          block,
          std::min(32, n_tokens - (int)spec.size()),
          widths > 1 ? (uint32_t)width : 5u,
          &produced,
          nullptr,
          false};
      if (ds41_sessions_dspark_step_batch(&item, 1, error, sizeof(error)) !=
          0) {
        std::fprintf(stderr, "DSpark step failed: %s\n", error);
        return 1;
      }
      spec.insert(spec.end(), block, block + produced);
    }
    const double spec_ms = ms(t0, now());
    uint64_t drafted = 0, accepted = 0, steps = 0, full = 0;
    ds41_session_dspark_stats(session, nullptr, nullptr, &drafted, &accepted,
                              nullptr, nullptr, &full, &steps, nullptr,
                              nullptr);
    drafted -= base_drafted;
    accepted -= base_accepted;
    steps -= base_steps;
    full -= base_full;
    if (widths > 1)
      std::printf("width %d: ", width);
    int same = 0;
    while (same < n_tokens && plain[same] == spec[same])
      same++;
    std::printf("plain %d tokens %.1f ms (%.2f tok/s)\n", n_tokens, plain_ms,
                n_tokens * 1000.0 / plain_ms);
    std::printf(
        "dspark %zu tokens %.1f ms (%.2f tok/s), %llu cycles, drafted %llu "
        "accepted %llu (%.1f%%), %.2f "
        "tokens/cycle, %llu full blocks\n",
        spec.size(), spec_ms, spec.size() * 1000.0 / spec_ms,
        (unsigned long long)steps, (unsigned long long)drafted,
        (unsigned long long)accepted,
        drafted ? 100.0 * accepted / drafted : 0.0,
        steps ? (double)spec.size() / steps : 0.0, (unsigned long long)full);
    std::printf("greedy match: %d of %d tokens%s\n", same, n_tokens,
                same == n_tokens ? " (identical)" : "");
    if (same < n_tokens) {
      std::printf("plain:");
      for (int i = same; i < std::min(n_tokens, same + 8); i++)
        std::printf(" %d", plain[i]);
      std::printf("\ndspark:");
      for (int i = same; i < std::min(n_tokens, same + 8); i++)
        std::printf(" %d", spec[i]);
      std::printf("\n");
    }
    if (same < n_tokens)
      return 1;
  }
  ds41_session_free(session);
  return 0;
}

/* --dspark-dump DIR: prefills the prompt, drafts after its argmax and writes
 * the drafter inputs and outputs for tools/ds41/dspark_reference.py:
 * features.f32 ([rows][3 * dim] target features of the last rows),
 * logits.f32 ([block][vocab] head logits before the Markov bias) and
 * draft.txt (first feature position, rows, anchor, drafts). */
int DsparkDump(ds41_engine* engine, const std::vector<int>& prompt,
               const char* dir) {
  ds41_decoder* d = ds41_decoder_create(engine, (uint32_t)prompt.size() + 16);
  ds41_dspark_state* s = d ? ds41_dspark_state_create(engine, d) : nullptr;
  std::vector<float> logits((std::size_t)ds41_engine_vocab_size(engine));
  if (!s)
    return 1;
  const uint32_t cap = ds41_decoder_batch_capacity(d);
  uint32_t pos0 = 0, rows = 0;
  const float* features = nullptr;
  for (uint32_t at = 0; at < prompt.size();) {
    const uint32_t n = std::min<uint32_t>(cap, (uint32_t)prompt.size() - at);
    if (ds41_decoder_eval(d, prompt.data() + at, n, logits.data(), 1, nullptr,
                          nullptr) != 0)
      return 1;
    features = ds41_decoder_features(d, &pos0, &rows);
    /* The ring holds only the dumped rows, so the oracle sees the same keys. */
    ds41_dspark_state_reset(s, pos0);
    if (!ds41_dspark_inject(s, pos0, rows))
      return 1;
    at += n;
  }
  const int anchor = Argmax(logits.data(), logits.size());
  int32_t drafts[DS41_DSPARK_BLOCK];
  if (!ds41_dspark_draft(s, anchor, drafts))
    return 1;
  std::vector<float> host((std::size_t)rows * DS41_DSPARK_TARGET_LAYERS * 5120);
  std::vector<float> draft_logits((std::size_t)DS41_DSPARK_BLOCK *
                                  logits.size());
  if (hipMemcpy(host.data(), features, host.size() * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess ||
      !ds41_dspark_copy_logits(engine, draft_logits.data()))
    return 1;
  const std::string base(dir);
  std::ofstream(base + "/features.f32", std::ios::binary)
      .write((const char*)host.data(), host.size() * 4);
  std::ofstream(base + "/logits.f32", std::ios::binary)
      .write((const char*)draft_logits.data(), draft_logits.size() * 4);
  std::ofstream meta(base + "/draft.txt");
  meta << pos0 << " " << rows << " " << anchor;
  for (int32_t t : drafts)
    meta << " " << t;
  meta << "\n";
  std::printf("dumped %u feature rows from %u, anchor %d, drafts", rows, pos0,
              anchor);
  for (int32_t t : drafts)
    std::printf(" %d", t);
  std::printf("\n");
  ds41_dspark_state_destroy(s);
  ds41_decoder_destroy(d);
  return 0;
}

int main(int argc, char** argv) {
  const char* model = nullptr;
  const char* engram = nullptr;
  const char* tokenize = nullptr;
  const char* decode = nullptr;
  const char* out_dir = nullptr;
  int generate = 0;
  uint32_t context = 0;
  uint32_t prefill_batch = 0;
  bool score = false;
  bool batch = false;
  bool upload = false;
  bool rendered = false;
  int spec_check = 0;
  int spec_bench = -1;
  const char* dspark = nullptr;
  int dspark_generate = 0;
  const char* dspark_dump = nullptr;
  bool dspark_sweep = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--engram") == 0 && i + 1 < argc) {
      engram = argv[++i];
    } else if (std::strcmp(argv[i], "--tokenize") == 0 && i + 1 < argc) {
      tokenize = argv[++i];
    } else if (std::strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
      decode = argv[++i];
      upload = true;
    } else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
      generate = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
      context = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--prefill-batch") == 0 && i + 1 < argc) {
      prefill_batch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--score") == 0) {
      score = true;
    } else if (std::strcmp(argv[i], "--batch") == 0) {
      batch = true;
    } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--spec-check") == 0 && i + 1 < argc) {
      spec_check = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--spec-bench") == 0 && i + 1 < argc) {
      spec_bench = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--dspark") == 0 && i + 1 < argc) {
      dspark = argv[++i];
    } else if (std::strcmp(argv[i], "--dspark-generate") == 0 && i + 1 < argc) {
      dspark_generate = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--dspark-sweep") == 0) {
      dspark_sweep = true;
    } else if (std::strcmp(argv[i], "--dspark-dump") == 0 && i + 1 < argc) {
      dspark_dump = argv[++i];
      upload = true;
    } else if (std::strcmp(argv[i], "--rendered") == 0) {
      rendered = true;
    } else if (std::strcmp(argv[i], "--upload") == 0) {
      upload = true;
    } else if (!model) {
      model = argv[i];
    } else {
      model = nullptr;
      break;
    }
  }
  if (!model) {
    std::fprintf(
        stderr,
        "usage: %s MODEL-00001-of-N.gguf [--engram FILE] [--upload] "
        "[--tokenize TEXTS [--rendered]] [--decode IDS [--generate N] "
        "[--score] [--batch] [--context N] [--prefill-batch N] "
        "[--out DIR]] [--spec-check CYCLES | --spec-bench N | --dspark FILE "
        "--dspark-generate N [--dspark-sweep] | --dspark FILE --dspark-dump "
        "DIR]\n",
        argv[0]);
    return 2;
  }
  Report("start");
  const auto begin = std::chrono::steady_clock::now();
  ds41_engine* engine = nullptr;
  ds41_engine_options options{};
  options.model_path = model;
  options.engram_path = engram;
  options.validate_only = !upload;
  options.prefill_batch = prefill_batch;
  options.dspark_model_path = dspark;
  if (ds41_engine_open(&engine, &options) != 0)
    return 1;
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  std::printf("opened %s in %.1f s (vocab %d)\n",
              ds41_engine_model_name(engine), seconds,
              ds41_engine_vocab_size(engine));
  Report("loaded");
  if (tokenize) {
    /* One text per line; "\n" in a line stands for a newline. Prints the ids.
     */
    std::ifstream in(tokenize);
    std::string line;
    while (std::getline(in, line)) {
      std::string text;
      for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') {
          text += '\n';
          ++i;
        } else {
          text += line[i];
        }
      }
      ds41_tokens tokens{};
      if (rendered)
        ds41_tokenize_rendered_chat(engine, text.c_str(), &tokens);
      else
        ds41_tokenize_text(engine, text.c_str(), &tokens);
      std::printf("ids");
      for (int i = 0; i < tokens.len; ++i)
        std::printf(" %d", tokens.v[i]);
      std::printf("\n");
      ds41_tokens_free(&tokens);
    }
  }
  if (decode &&
      (spec_check || spec_bench >= 0 || dspark_generate > 0 || dspark_dump)) {
    std::vector<int> prompt;
    for (const char* p = decode; *p;) {
      prompt.push_back(std::atoi(p));
      while (*p && *p != ',')
        ++p;
      if (*p == ',')
        ++p;
    }
    if (dspark_dump)
      return DsparkDump(engine, prompt, dspark_dump);
    if (dspark_generate > 0)
      return DsparkProbe(engine, prompt, context, dspark_generate,
                         dspark_sweep);
    return SpeculativeProbe(engine, prompt, context, spec_check, spec_bench);
  }
  if (decode) {
    /* Feed comma-separated ids one at a time (--batch: in decoder batches),
     * then decode greedily. */
    std::vector<int> tokens;
    for (const char* p = decode; *p;) {
      tokens.push_back(std::atoi(p));
      while (*p && *p != ',')
        ++p;
      if (*p == ',')
        ++p;
    }
    const uint32_t total = (uint32_t)tokens.size() + (uint32_t)generate;
    ds41_decoder* decoder =
        ds41_decoder_create(engine, std::max(context, total + 1));
    if (!decoder)
      return 1;
    Report("decoder");
    Dump dump;
    if (out_dir) {
      dump.dir = out_dir;
      dump.routes = std::fopen((dump.dir + "/routes.txt").c_str(), "w");
    }
    /* --score writes the argmax and next-token NLL per prompt position instead
     * of logits. */
    FILE* logits_file =
        out_dir && !score ? std::fopen((dump.dir + "/logits.f32").c_str(), "wb")
                          : nullptr;
    FILE* score_file = out_dir && score
                           ? std::fopen((dump.dir + "/score.txt").c_str(), "w")
                           : nullptr;
    const size_t vocab = (size_t)ds41_engine_vocab_size(engine);
    std::vector<float> logits(vocab);
    std::vector<int> generated;
    /* Records the logits of position `step` (input `token`). */
    auto consume = [&](uint32_t step, int token, const float* row, double ms) {
      if (logits_file)
        std::fwrite(row, sizeof(float), vocab, logits_file);
      int best = 0;
      for (size_t i = 1; i < vocab; ++i)
        if (row[i] > row[(size_t)best])
          best = (int)i;
      if (step + 1 >= tokens.size())
        generated.push_back(best);
      if (score_file && step + 1 < tokens.size()) {
        double sum = 0.0;
        for (size_t i = 0; i < vocab; ++i)
          sum += std::exp((double)row[i] - (double)row[(size_t)best]);
        const double nll =
            std::log(sum) -
            ((double)row[(size_t)tokens[step + 1]] - (double)row[(size_t)best]);
        std::fprintf(score_file, "%u %d %d %.6f %.2f\n", step, best,
                     tokens[step + 1], nll, ms);
      } else if (!score_file) {
        std::printf("step %u token %d -> %d (%.1f ms)\n", step, token, best,
                    ms);
      }
    };
    uint32_t step = 0;
    if (batch) {
      /* The prompt in decoder-sized batches with logits for every position. */
      const uint32_t cap = ds41_decoder_batch_capacity(decoder);
      std::vector<float> rows((size_t)cap * vocab);
      while (step < tokens.size()) {
        const uint32_t n =
            std::min<uint32_t>(cap, (uint32_t)tokens.size() - step);
        const auto t0 = std::chrono::steady_clock::now();
        /* Every row's logits only when they are recorded. */
        const uint32_t logit_rows = logits_file || score_file ? n : 1;
        if (ds41_decoder_eval(decoder, tokens.data() + step, n, rows.data(),
                              logit_rows, nullptr, nullptr) != 0) {
          std::fprintf(stderr, "batch failed at step %u\n", step);
          return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        std::fprintf(stderr, "batch %u tokens at %u: %.1f ms (%.1f tok/s)\n", n,
                     step, ms, n * 1000.0 / ms);
        for (uint32_t i = n - logit_rows; i < n; ++i)
          consume(step + i, tokens[step + i],
                  rows.data() + (size_t)(i - (n - logit_rows)) * vocab, ms / n);
        step += n;
      }
    }
    for (; step < total; ++step) {
      const int token = step < tokens.size() ? tokens[step] : generated.back();
      dump.pos = step;
      const auto t0 = std::chrono::steady_clock::now();
      if (ds41_decoder_step(decoder, token, logits.data(),
                            out_dir && !score ? DumpLayer : nullptr,
                            &dump) != 0) {
        std::fprintf(stderr, "decode failed at step %u\n", step);
        return 1;
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      consume(step, token, logits.data(), ms);
    }
    std::printf("generated");
    for (size_t i = 0;
         i + 1 < generated.size() || (i < generated.size() && generate > 0);
         ++i)
      std::printf(" %d", generated[i]);
    std::printf("\n");
    if (logits_file)
      std::fclose(logits_file);
    if (score_file)
      std::fclose(score_file);
    if (dump.routes)
      std::fclose(dump.routes);
    ds41_decoder_destroy(decoder);
  }
  ds41_engine_close(engine);
  Report("closed");
  return 0;
}
