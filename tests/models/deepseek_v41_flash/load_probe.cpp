/* Opens a DeepSeek V4.1 Flash build and reports its memory plan. With
 * --upload the resident weights are copied to the GPU and the memory peak is
 * reported; otherwise the artifacts are only parsed, validated and bound. */
#include <glob.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

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
  bool upload = false;
  bool rendered = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--engram") == 0 && i + 1 < argc) {
      engram = argv[++i];
    } else if (std::strcmp(argv[i], "--tokenize") == 0 && i + 1 < argc) {
      tokenize = argv[++i];
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
                 "[--tokenize TEXTS [--rendered]]\n",
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
  ds41_engine_close(engine);
  Report("closed");
  return 0;
}
