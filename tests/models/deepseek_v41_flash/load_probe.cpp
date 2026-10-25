/* Opens a DeepSeek V4.1 Flash build and reports its memory plan. With
 * --upload the resident weights are copied to the GPU and the memory peak is
 * reported; otherwise the artifacts are only parsed, validated and bound. */
#include <glob.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/model.h"

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
}  // namespace

int main(int argc, char** argv) {
  const char* model = nullptr;
  const char* engram = nullptr;
  const char* tokenize = nullptr;
  const char* decode = nullptr;
  const char* out_dir = nullptr;
  int generate = 0;
  uint32_t context = 0;
  bool score = false;
  bool batch = false;
  bool upload = false;
  bool rendered = false;
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
    } else if (std::strcmp(argv[i], "--score") == 0) {
      score = true;
    } else if (std::strcmp(argv[i], "--batch") == 0) {
      batch = true;
    } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_dir = argv[++i];
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
    std::fprintf(stderr,
                 "usage: %s MODEL-00001-of-N.gguf [--engram FILE] [--upload] "
                 "[--tokenize TEXTS [--rendered]] [--decode IDS [--generate N] "
                 "[--score] [--batch] [--context N] [--out DIR]]\n",
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
