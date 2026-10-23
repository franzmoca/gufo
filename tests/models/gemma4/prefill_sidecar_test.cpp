// Model-backed check of the int4 prefill copies (--prefill-sidecar): decode
// and small-batch rows keep the GGUF arithmetic (scalar reference limits of
// gemma4.target), and prefill rows stay inside the envelope measured at
// introduction, on a short prompt against the scalar reference and on the
// long conversation against the exact small-batch rows.
// Requires GUFO_GEMMA4_MODEL and GUFO_GEMMA4_PREFILL_SIDECAR; exits 77
// without them.
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/reference.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/model_logits.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Compare;
using gemma4_test::ConversationTokens;
using gemma4_test::GpuLogits;
using gemma4_test::PromptTokens;
using gemma4_test::Require;
using gemma4_test::Stats;

namespace {

/// Decode keeps the GGUF limit of gemma4.target. The int4 limits were set at
/// introduction (2026-10-23, gemma-4-31B-it UD-Q4_K_XL with the
/// calibration-v6 sidecar) at about twice the observed values: prefill 0.50
/// (top-1 20/22), the conversation 1.15 (1047/1542), decoding its last 256
/// tokens after an int4 prefill 0.10 (234/256). The conversation is a hard
/// case for any quantization (BF16 perplexity 538; UD-Q4_K_XL against BF16
/// 0.36, the simulated int4 copy against BF16 1.18, top-1 0.67).
constexpr double kDecode = 1e-5;
constexpr double kPrefill = 1.0;
constexpr double kLongPrefill = 2.3;
constexpr double kDecodeAfterPrefill = 0.2;

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MODEL");
  const char* sidecar = std::getenv("GUFO_GEMMA4_PREFILL_SIDECAR");
  if (path == nullptr || *path == '\0' || sidecar == nullptr ||
      *sidecar == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL or GUFO_GEMMA4_PREFILL_SIDECAR is "
                 "not set\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    options.max_context = 4096;
    options.prefill_sidecar = sidecar;
    auto model = g4::Model::Load(path, options, &error);
    Require(model != nullptr, error);
    const std::size_t vocab = model->VocabSize();
    const auto tokens =
        PromptTokens(*model, "Write one sentence about the sea near Genoa.");
    Require(tokens.size() > 16, "prompt too short for the prefill path");

    const auto weights = g4::ModelWeights::Bind(model->reader(), &error);
    Require(weights.has_value(), error);
    g4::Reference reference(*weights, g4::Reference::Storage::kHalfKv);
    std::vector<float> oracle;
    reference.Forward(tokens, &oracle, nullptr);

    const Stats d = Compare(oracle, GpuLogits(*model, tokens, 1), vocab);
    const Stats c = Compare(oracle, GpuLogits(*model, tokens, 8), vocab);
    const Stats p =
        Compare(oracle, GpuLogits(*model, tokens, tokens.size()), vocab);
    std::cout << "decode:  mean KL " << d.mean_kl << ", top-1 " << d.top1 << "/"
              << d.rows << '\n'
              << "8-row:   mean KL " << c.mean_kl << ", top-1 " << c.top1 << "/"
              << c.rows << '\n'
              << "prefill: mean KL " << p.mean_kl << ", max " << p.max_kl
              << ", top-1 " << p.top1 << "/" << p.rows << '\n';
    Require(d.mean_kl < kDecode && d.top1 == d.rows,
            "decode diverges from the reference");
    Require(c.mean_kl < kDecode && c.top1 == c.rows,
            "small-batch rows diverge from the reference");
    Require(p.mean_kl < kPrefill, "int4 prefill exceeds its KL envelope");

    const auto conversation = ConversationTokens(*model);
    const auto exact = GpuLogits(*model, conversation, 8);
    const auto bulk = GpuLogits(*model, conversation, conversation.size());
    const Stats lp = Compare(exact, bulk, vocab);
    std::cout << "long prefill vs exact rows (" << lp.rows
              << " tokens): mean KL " << lp.mean_kl << ", max " << lp.max_kl
              << ", top-1 " << lp.top1 << "/" << lp.rows << '\n';
    Require(lp.mean_kl < kLongPrefill,
            "int4 long prefill exceeds its KL envelope");

    // Deployment: the tail of the conversation decoded in small-batch rows
    // (GGUF weights) after an int4 prefill of its head, against exact rows
    // throughout.
    constexpr std::size_t kTail = 256;
    const std::size_t head = conversation.size() - kTail;
    auto session = model->CreateSession(0, &error);
    Require(session != nullptr, error);
    std::vector<float> part;
    Require(session->EvaluateAll(std::span(conversation).first(head), &part,
                                 &error),
            error);
    std::vector<float> tail;
    for (std::size_t at = head; at < conversation.size(); at += 8) {
      Require(session->EvaluateAll(std::span(conversation).subspan(at, 8),
                                   &part, &error),
              error);
      tail.insert(tail.end(), part.begin(), part.end());
    }
    const std::vector<float> exact_tail(exact.end() - kTail * vocab,
                                        exact.end());
    const Stats dt = Compare(exact_tail, tail, vocab);
    std::cout << "decode after int4 prefill (" << dt.rows
              << " tokens): mean KL " << dt.mean_kl << ", max " << dt.max_kl
              << ", top-1 " << dt.top1 << "/" << dt.rows << '\n';
    Require(dt.mean_kl < kDecodeAfterPrefill,
            "decoding after an int4 prefill exceeds its KL envelope");
  });
}
