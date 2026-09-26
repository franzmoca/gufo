// Evaluates a token file on the GPU runtime and writes teacher-forced logits
// for comparison with tools/gemma4/llama_logits and the CPU reference:
//   gemma4_gpu_probe --model GGUF --tokens T.i32 --logits-out L.g4lg
//       [--chunk N]   (tokens per EvaluateAll call; default: all at once)
//       [--first N --stride N]  (keep only these rows in the output)
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/gemma4/engine.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, tokens_path, logits_out;
  std::size_t chunk = 0;
  std::size_t first = 0;
  std::size_t stride = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        std::exit(2);
      return argv[++i];
    };
    if (arg == "--model")
      model = value();
    else if (arg == "--tokens")
      tokens_path = value();
    else if (arg == "--logits-out")
      logits_out = value();
    else if (arg == "--chunk")
      chunk = std::stoul(value());
    else if (arg == "--first")
      first = std::stoul(value());
    else if (arg == "--stride")
      stride = std::stoul(value());
    else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    Require(!model.empty() && !tokens_path.empty() && !logits_out.empty(),
            "usage: --model GGUF --tokens T.i32 --logits-out L [--chunk N]");
    std::ifstream in(tokens_path, std::ios::binary);
    std::vector<g4::TokenId> tokens;
    for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), 4);) {
      tokens.push_back(t);
    }
    Require(!tokens.empty(), "no tokens");
    std::string error;
    g4::ModelOptions options;
    options.max_context = static_cast<std::uint32_t>(tokens.size() + 256);
    const auto start = std::chrono::steady_clock::now();
    auto m = g4::Model::Load(model, options, &error);
    Require(m != nullptr, error);
    const auto loaded = std::chrono::steady_clock::now();
    auto session = m->CreateSession(0, &error);
    Require(session != nullptr, error);
    gemma4_test::LogitFile file;
    file.vocab = m->VocabSize();
    const std::size_t step = chunk == 0 ? tokens.size() : chunk;
    for (std::size_t begin = 0; begin < tokens.size(); begin += step) {
      const std::size_t count = std::min(step, tokens.size() - begin);
      std::vector<float> logits;
      Require(session->EvaluateAll(std::span(tokens).subspan(begin, count),
                                   &logits, &error),
              error);
      for (std::size_t r = 0; r < count; ++r) {
        const std::size_t p = begin + r;
        if (p >= first && (p - first) % stride == 0) {
          file.positions.push_back(static_cast<std::uint32_t>(p));
          file.logits.insert(file.logits.end(), logits.begin() + r * file.vocab,
                             logits.begin() + (r + 1) * file.vocab);
        }
      }
    }
    const auto done = std::chrono::steady_clock::now();
    gemma4_test::WriteLogits(logits_out, file);
    std::cout << "load "
              << std::chrono::duration<double>(loaded - start).count() << " s, "
              << tokens.size() << " tokens in "
              << std::chrono::duration<double>(done - loaded).count() << " s\n";
  });
}
