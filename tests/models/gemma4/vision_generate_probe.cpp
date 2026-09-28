// Image prompt probe: greedy generation from an image question, then the
// teacher-forced logits of every position from the image's closing token on,
// for comparison with tools/gemma4/llama_vision.cpp on the same tokens.
//
//   gemma4_vision_generate_probe --model GGUF --mmproj GGUF --image FILE
//       [--question TEXT] [--tokens N] [--out-dir DIR]
// DIR receives tokens.i32, image<i>.rgb (resized pixels; sizes printed) and
// gufo.g4lg.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
namespace vision = gufo::models::gemma4::vision;
using gemma4_test::Require;

namespace {

std::size_t Argmax(std::span<const float> logits) {
  return static_cast<std::size_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void Run(int argc, char** argv) {
  std::string model_path, mmproj, image_path, out_dir;
  std::string question = "Describe this image in detail.";
  std::size_t count = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--model")
      model_path = argv[i + 1];
    else if (arg == "--mmproj")
      mmproj = argv[i + 1];
    else if (arg == "--image")
      image_path = argv[i + 1];
    else if (arg == "--question")
      question = argv[i + 1];
    else if (arg == "--tokens")
      count = std::stoul(argv[i + 1]);
    else if (arg == "--out-dir")
      out_dir = argv[i + 1];
    else
      throw std::runtime_error("unknown option " + arg);
  }
  Require(!model_path.empty() && !mmproj.empty() && !image_path.empty(),
          "--model, --mmproj and --image are required");
  std::string error;
  g4::ModelOptions options;
  options.max_context = 8192;
  auto model = g4::Model::Load(model_path, options, &error);
  Require(model != nullptr, error);
  vision::Encoder encoder(mmproj, model->config().hidden_size);

  std::vector<gufo::tokenization::ChatMessage> messages(1);
  messages[0].role = gufo::tokenization::ChatRole::kUser;
  messages[0].content = question;
  messages[0].images.push_back(
      {0, std::make_shared<const std::vector<std::uint8_t>>(
              gufo::core::ReadImageFile(image_path))});
  const auto prompt = vision::Prepare(model->tokenizer(), messages, {}, {},
                                      encoder.identity(), options.max_context);
  std::vector<std::shared_ptr<const vision::Encoder::Embedding>> embeddings;
  std::vector<g4::ImageSpan> spans;
  for (const auto& image : prompt.images) {
    const auto start = std::chrono::steady_clock::now();
    embeddings.push_back(encoder.Encode(image.pixels));
    std::cerr << "image " << image.pixels.width << "x" << image.pixels.height
              << ": " << image.span.rows << " rows, encoded in "
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << " ms\n";
    spans.push_back(
        {image.span.offset, image.span.rows, image.prefix_identity});
  }
  const auto embed = [&](std::size_t i) { return embeddings.at(i)->data(); };

  auto session = model->CreateSession(options.max_context, &error);
  Require(session != nullptr, error);
  const auto start = std::chrono::steady_clock::now();
  Require(session->Sync(prompt.tokens, spans, embed, &error), error);
  std::cerr << "prefill " << prompt.tokens.size() << " tokens: "
            << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count()
            << " ms\n";
  std::vector<g4::TokenId> tokens = prompt.tokens;
  std::vector<g4::TokenId> generated;
  for (std::size_t i = 0; i < count; ++i) {
    const auto token = static_cast<g4::TokenId>(Argmax(session->Logits()));
    if (model->IsStopToken(token)) {
      break;
    }
    generated.push_back(token);
    tokens.push_back(token);
    Require(session->Evaluate(token, &error), error);
  }
  std::cout << model->Decode(generated) << '\n';
  if (out_dir.empty()) {
    return;
  }

  // Teacher-forced rows from the last image's closing token on.
  const auto& last = prompt.images.back().span;
  const std::size_t first = last.offset + last.rows + 1;
  auto fresh = model->CreateSession(options.max_context, &error);
  Require(fresh != nullptr, error);
  Require(fresh->Sync(std::span(tokens).first(first), spans, embed, &error),
          error);
  gemma4_test::LogitFile file;
  file.vocab = model->VocabSize();
  file.positions.push_back(static_cast<std::uint32_t>(first - 1));
  file.logits.assign(fresh->Logits().begin(), fresh->Logits().end());
  std::vector<float> rows;
  Require(fresh->EvaluateAll(std::span(tokens).subspan(first), &rows, &error),
          error);
  for (std::size_t p = first; p < tokens.size(); ++p) {
    file.positions.push_back(static_cast<std::uint32_t>(p));
  }
  file.logits.insert(file.logits.end(), rows.begin(), rows.end());
  gemma4_test::WriteLogits(out_dir + "/gufo.g4lg", file);
  gemma4_test::WriteTokens(out_dir + "/tokens.i32", tokens);
  for (std::size_t i = 0; i < prompt.images.size(); ++i) {
    const auto& pixels = prompt.images[i].pixels;
    std::ofstream out(out_dir + "/image" + std::to_string(i) + ".rgb",
                      std::ios::binary);
    out.write(reinterpret_cast<const char*>(pixels.pixels.data()),
              static_cast<std::streamsize>(pixels.pixels.size()));
    std::cerr << "image" << i << ".rgb " << pixels.width << ' ' << pixels.height
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  return gemma4_test::Run([&] { Run(argc, argv); });
}
