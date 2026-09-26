// Runs the scalar Gemma 4 reference on a prompt and writes its tokens and
// logits for comparison with tools/gemma4/llama_logits and the GPU runtime.
//   gemma4_reference_probe --model GGUF (--text TEXT | --chat MESSAGE)
//       --tokens-out T.i32 --logits-out L.g4lg [--half-kv] [--chunk N]
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/reference.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, text, chat, tokens_out, logits_out, trace_out;
  bool half_kv = false;
  bool q8 = false;
  std::size_t chunk = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        std::exit(2);
      return argv[++i];
    };
    if (arg == "--model")
      model = value();
    else if (arg == "--text")
      text = value();
    else if (arg == "--chat")
      chat = value();
    else if (arg == "--tokens-out")
      tokens_out = value();
    else if (arg == "--logits-out")
      logits_out = value();
    else if (arg == "--half-kv")
      half_kv = true;
    else if (arg == "--q8-activations")
      q8 = true;
    else if (arg == "--trace-out")
      trace_out = value();
    else if (arg == "--chunk")
      chunk = std::stoul(value());
    else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    Require(!model.empty() && !tokens_out.empty() && !logits_out.empty() &&
                (text.empty() != chat.empty()),
            "usage: --model GGUF (--text T | --chat M) --tokens-out F "
            "--logits-out F [--half-kv] [--chunk N]");
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Require(reader != nullptr, error);
    const auto weights = g4::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), error);
    const auto tokenizer = g4::Tokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, error);

    std::vector<g4::TokenId> tokens;
    if (!chat.empty()) {
      gufo::tokenization::ChatMessage message(
          gufo::tokenization::ChatRole::kUser, chat);
      const auto rendered =
          g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
      Require(rendered.has_value(), error);
      tokens = tokenizer->Encode(rendered->text, false, true);
    } else {
      tokens = tokenizer->Encode(text, true, true);
    }
    gemma4_test::WriteTokens(tokens_out, tokens);

    g4::Reference reference(*weights,
                            q8        ? g4::Reference::Storage::kQ8Activations
                            : half_kv ? g4::Reference::Storage::kHalfKv
                                      : g4::Reference::Storage::kFloat32);
    gemma4_test::LogitFile file;
    file.vocab = weights->vocab_size;
    const std::size_t step = chunk == 0 ? tokens.size() : chunk;
    Require(trace_out.empty() || step == tokens.size(),
            "--trace-out needs a single chunk");
    reference.SetTrace(!trace_out.empty());
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t begin = 0; begin < tokens.size(); begin += step) {
      const std::size_t count = std::min(step, tokens.size() - begin);
      std::vector<float> logits;
      reference.Forward(std::span(tokens).subspan(begin, count), &logits,
                        nullptr);
      file.logits.insert(file.logits.end(), logits.begin(), logits.end());
    }
    for (std::size_t p = 0; p < tokens.size(); ++p) {
      file.positions.push_back(static_cast<std::uint32_t>(p));
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    gemma4_test::WriteLogits(logits_out, file);
    if (!trace_out.empty()) {
      std::ofstream t(trace_out, std::ios::binary);
      const std::uint32_t th[3] = {weights->config.num_layers,
                                   static_cast<std::uint32_t>(tokens.size()),
                                   weights->config.hidden_size};
      t.write(reinterpret_cast<const char*>(th), sizeof(th));
      const auto& trace = reference.LayerTrace();
      t.write(reinterpret_cast<const char*>(trace.data()),
              static_cast<std::streamsize>(trace.size() * 4));
    }
    std::cout << tokens.size() << " tokens in " << seconds << " s\n";
  });
}
