// Model-backed GPU qualification of the Gemma 4 target against the scalar
// reference on the same GGUF: decode arithmetic (FP32 activations) must be
// essentially exact, prefill (Q8_1 activations) must stay inside the
// envelope measured at introduction, and session prefix extension must be
// bitwise stable. Requires GUFO_GEMMA4_MODEL; exits 77 without it.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/reference.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

/// Limits recorded when the runtime was introduced (2026-09-26); they are
/// never loosened. Observed: decode mean KL 2.4e-7, prefill mean KL 0.0084.
constexpr double kDecodeMeanKl = 1e-5;
constexpr double kPrefillMeanKl = 0.015;

struct Stats {
  double mean_kl{0.0};
  double max_kl{0.0};
  std::size_t top1{0};
  std::size_t rows{0};
};

Stats Compare(const std::vector<float>& ref, const std::vector<float>& got,
              std::size_t vocab) {
  Stats s;
  s.rows = ref.size() / vocab;
  for (std::size_t r = 0; r < s.rows; ++r) {
    const float* a = &ref[r * vocab];
    const float* b = &got[r * vocab];
    double ma = -INFINITY, mb = -INFINITY;
    std::size_t aa = 0, ab = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      Require(std::isfinite(b[i]), "non-finite GPU logit");
      if (a[i] > ma) {
        ma = a[i];
        aa = i;
      }
      if (b[i] > mb) {
        mb = b[i];
        ab = i;
      }
    }
    double za = 0.0, zb = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      za += std::exp(a[i] - ma);
      zb += std::exp(b[i] - mb);
    }
    const double la = std::log(za) + ma, lb = std::log(zb) + mb;
    double kl = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      const double pa = a[i] - la;
      kl += std::exp(pa) * (pa - (b[i] - lb));
    }
    s.mean_kl += kl / static_cast<double>(s.rows);
    s.max_kl = std::max(s.max_kl, kl);
    s.top1 += aa == ab ? 1 : 0;
  }
  return s;
}

std::vector<float> GpuLogits(g4::Model& model,
                             const std::vector<g4::TokenId>& tokens,
                             std::size_t chunk) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session != nullptr, error);
  std::vector<float> all, part;
  for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {
    const std::size_t count = std::min(chunk, tokens.size() - begin);
    Require(session->EvaluateAll(std::span(tokens).subspan(begin, count), &part,
                                 &error),
            error);
    all.insert(all.end(), part.begin(), part.end());
  }
  return all;
}

/// Greedy tokens from autoregressive evaluation and from DecodeStep.
std::vector<g4::TokenId> Generate(g4::Model& model,
                                  std::span<const g4::TokenId> prompt,
                                  std::size_t count, bool speculative) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session && session->Sync(prompt, &error), error);
  gufo::sampling::SamplingConfig greedy;
  greedy.temperature = 0.0F;
  gufo::sampling::SamplerState sampler(greedy, {});
  std::vector<g4::TokenId> out;
  if (!speculative) {
    while (out.size() < count) {
      const auto token =
          static_cast<g4::TokenId>(sampler.Sample(session->Logits()));
      sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
      out.push_back(token);
      Require(session->Evaluate(token, &error), error);
    }
    return out;
  }
  while (out.size() < count) {
    g4::Session::DecodeResult step;
    Require(session->DecodeStep(count - out.size(), sampler, &step, &error,
                                false) &&
                !step.tokens.empty(),
            error);
    out.insert(out.end(), step.tokens.begin(), step.tokens.end());
  }
  const auto& stats = session->Statistics();
  std::cout << "speculative: cycles " << stats.cycles << ", drafted "
            << stats.drafted << ", accepted " << stats.accepted << '\n';
  return out;
}

std::vector<g4::TokenId> PromptTokens(const g4::Model& model,
                                      const std::string& text) {
  gufo::tokenization::ChatMessage message(gufo::tokenization::ChatRole::kUser,
                                          text);
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL is not set\n";
    return 77;
  }
  const char* draft = std::getenv("GUFO_GEMMA4_MTP_MODEL");
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    options.max_context = 1024;
    auto model = g4::Model::Load(path, options, &error);
    Require(model != nullptr, error);
    const std::size_t vocab = model->VocabSize();
    const auto tokens =
        PromptTokens(*model, "Write one sentence about the sea near Genoa.");
    Require(tokens.size() > 16, "prompt too short for the prefill path");

    // Independent scalar reference over the same weights.
    const auto weights = g4::ModelWeights::Bind(model->reader(), &error);
    Require(weights.has_value(), error);
    g4::Reference reference(*weights, g4::Reference::Storage::kHalfKv);
    std::vector<float> oracle;
    reference.Forward(tokens, &oracle, nullptr);

    const auto decode = GpuLogits(*model, tokens, 1);
    const auto chunked = GpuLogits(*model, tokens, 8);
    const auto prefill = GpuLogits(*model, tokens, tokens.size());
    const Stats d = Compare(oracle, decode, vocab);
    const Stats c = Compare(oracle, chunked, vocab);
    const Stats p = Compare(oracle, prefill, vocab);
    std::cout << "decode:  mean KL " << d.mean_kl << ", max " << d.max_kl
              << ", top-1 " << d.top1 << "/" << d.rows << '\n'
              << "8-row:   mean KL " << c.mean_kl << ", top-1 " << c.top1 << "/"
              << c.rows << '\n'
              << "prefill: mean KL " << p.mean_kl << ", max " << p.max_kl
              << ", top-1 " << p.top1 << "/" << p.rows << '\n';
    Require(d.mean_kl < kDecodeMeanKl && d.top1 == d.rows,
            "decode diverges from the reference");
    Require(c.mean_kl < kDecodeMeanKl && c.top1 == c.rows,
            "small-batch rows diverge from the reference");
    Require(p.mean_kl < kPrefillMeanKl, "prefill exceeds its KL envelope");

    // Extending a synced prompt matches evaluating the tokens one by one.
    const std::size_t split = tokens.size() - 4;
    const std::span<const g4::TokenId> head(tokens.data(), split);
    auto a = model->CreateSession(0, &error);
    auto b = model->CreateSession(0, &error);
    Require(a && b, error);
    Require(a->Sync(head, &error) && b->Sync(head, &error), error);
    for (std::size_t i = split; i < tokens.size(); ++i) {
      Require(a->Evaluate(tokens[i], &error), error);
    }
    Require(b->Sync(tokens, &error), error);
    const auto as_vector = [](std::span<const float> v) {
      return std::vector<float>(v.begin(), v.end());
    };
    const Stats ext =
        Compare(as_vector(a->Logits()), as_vector(b->Logits()), vocab);
    Require(ext.top1 == 1 && ext.mean_kl < kDecodeMeanKl,
            "prefix extension diverges from token-by-token evaluation");
    // Rewinding to a shorter prompt reuses the prefix and matches a fresh sync.
    Require(b->Sync(head, &error), error);
    auto fresh = model->CreateSession(0, &error);
    Require(fresh && fresh->Sync(head, &error), error);
    const Stats rw =
        Compare(as_vector(fresh->Logits()), as_vector(b->Logits()), vocab);
    Require(rw.top1 == 1 && rw.mean_kl < kPrefillMeanKl,
            "rewound session diverges from a fresh one");

    if (draft == nullptr || *draft == '\0') {
      std::cout
          << "note: GUFO_GEMMA4_MTP_MODEL not set; speculation unchecked\n";
      return;
    }
    model.reset();
    options.mtp_model_path = draft;
    options.draft_tokens = 4;
    auto mtp = g4::Model::Load(path, options, &error);
    Require(mtp != nullptr, error);
    // With a drafter, verification rows round exactly like decode...
    Require(GpuLogits(*mtp, tokens, 1) == GpuLogits(*mtp, tokens, 5),
            "verification rows differ from single-token decode");
    // ...so greedy speculation reproduces autoregressive output.
    for (const char* text :
         {"Write one sentence about the sea near Genoa.",
          "List the first twelve prime numbers, separated by commas.",
          "Explain in three sentences why the sky is blue."}) {
      const auto prompt = PromptTokens(*mtp, text);
      const auto ar = Generate(*mtp, prompt, 64, false);
      const auto spec = Generate(*mtp, prompt, 64, true);
      Require(ar == spec, std::string("greedy MTP differs from AR: ") + text);
    }
  });
}
