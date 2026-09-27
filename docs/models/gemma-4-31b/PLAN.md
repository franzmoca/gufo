# Gemma 4 31B IT in Gufo: implementation plan

## Context

Gemma 4 31B IT is served today by `~/scripts/gemma-control.sh`. It runs llama-server from the
halo-box/strix-llama.cpp fork (8c1c282ec, Vulkan RADV, `GGML_VK_MMV_NO_SPLIT=1`, `-ub 512`,
f16 KV). For Unsloth's `gemma-4-31B-it-UD-Q4_K_XL.gguf` with its `MTP/mtp-gemma-4-31B-it-Q8_0.gguf`
drafter, `~/projects/gemma/results/STATUS.md` records:

- pp512 348 t/s and pp8192 307 t/s.
- AR tg128 11.40 t/s (10.8 t/s through the server).
- MTP n=4 20.2 t/s; n=2 19.6 t/s at 50% acceptance.
- Prefill 150–180 t/s at 20–33K depth, so a 33K prompt takes about 3.5 minutes.
- MTP2 falls to 9.1 t/s at 32K depth (measured on the Melinoe 31B fine-tune).

On Qwen3.8 27B, Gufo beats llama.cpp by 84–111% on prefill and 2.6% on AR decode, and its speculative decode is faster too.

**Goal:** a native gfx1151 Gemma 4 engine in Gufo, faster than the fork, at llama.cpp-level quality.
It must serve the same OpenAI-compatible workload: SillyTavern, contexts up to 131K, thinking
usually off, sampler temp 1 / top-k 64 / top-p 0.95 / repeat 1.05.

**Agreed scope:**
- **Text only.** Vision comes later.
- **Quality gates:** a scalar CPU oracle plus teacher-forced llama.cpp logits on the same GGUF.
- **Benchmarks:** against both the repo's pinned llama.cpp reference and the gemma-control fork.
- **Full HTTP parity:** reasoning channel, Gemma tool calls, RAM/disk prompt cache, MTP over HTTP.
- **Later:** fine-tunes and QAT (Q4_0). v1 rejects them with explicit errors.

## Architecture facts (verified from the GGUF headers and llama.cpp 391fac16 `src/models/gemma4*.cpp`)

The llama.cpp semantics were read from a local checkout at `391fac16`.

**Target model**

| Item | Value |
|---|---|
| Architecture / size | arch `gemma4`, 60 layers, hidden 5376, FFN 21504, 32 query heads, vocab 262144, native context 262144, eps 1e-6 |
| Global layers | Layers 5, 11, …, 59 (10 layers). hd512, 4 KV heads. Q 5376→16384, K 5376→2048, **no V tensor**: V = rms_noweight(raw K projection). O 16384→5376. NEOX RoPE, θ 1e6, 512 dims, frequencies divided by `rope_freqs[256]`: only the first 64 pairs (i, i+256) rotate (ff = 1e30 means identity) |
| Sliding layers | The other 50 layers. hd256, 16 KV heads. Q →8192, K/V →4096, O 8192→5376. NEOX RoPE, θ 1e4, all 256 dims. Window 1024: a key is masked when `pq − pk ≥ 1024` |

**Block, in order:**
1. `h = rms(x)·attn_norm`.
2. `Q = rope(rms(Wq h)·q_norm)`, `K = rope(rms(Wk h)·k_norm)`, `V = rms_noweight(Wv h | Kraw)`.
3. Attention with **scale 1.0**, no attention softcap.
4. `x += rms(Wo·a)·post_attention_norm`.
5. `x += rms(Wd(gelu_tanh(Wg h')·(Wu h')))·post_ffw_norm`, where `h' = rms(x)·ffn_norm`.
6. `x *= layer_output_scale[l]`, a scalar per layer.
7. RMSNorm is `x·rsqrt(mean+eps)·w` with no +1.

**Input and output:** the embedding is tied Q5_K `token_embd`, multiplied by `sqrtf(5376)`. At the end, `output_norm` produces `h_final`, the input to MTP. Then comes the tied LM head, then `30·tanh(l/30)`.

**Weights:** only Q4_K (11.86 GiB), Q6_K (3.04 GiB), Q5_K (2.62 GiB) and F32 tensors. Every K dimension is divisible by 256. Decode reads about 18.8 GB per token, so AR is capped near **12.8 t/s** at about 240 GB/s. llama.cpp already reaches 89% of that; the real wins are prefill, depth and MTP.

**Tokenizer** (`tokenizer.ggml.model=gemma4`):
- BPE on raw UTF-8 with ' '→U+2581; pre-split only on `[^\n]+|[\n]+`.
- Byte fallback `<0xXX>` (ids 238–493); ranks come from merge order (514,906 merges).
- User-defined tokens (always matched and printed): `<|tool_call>`48, `<tool_call|>`49, `<|tool_response>`50, `<tool_response|>`51, `<|"|>`52, `<|channel>`100, `<channel|>`101.
- Control tokens (hidden): bos 2, `<|turn>`105, `<turn|>`106, `<|think|>`98, `<|tool>`46, `<tool|>`47.
- EOG = {1, 106, 50}.

**Chat template:** this file carries Unsloth's variant, SHA-256 `845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`.
- It emits `<bos>` itself, so tokenize with `add_bos=false`.
- Thinking off appends `<|channel>thought\n<channel|>` to `<|turn>model\n`.
- Tool calls look like `<|tool_call>call:NAME{k:<|"|>str<|"|>,…}<tool_call|>`.
- It applies `trim`, `strip_thinking` on history, and a case-insensitive `dictsort`.

**MTP drafter** (`gemma4-assistant`, Q8_0, 0.5 GB):
- 4 layers, hidden 1024, SWA pattern [T,T,T,F], Q-only attention over the **target's** KV. Draft layers 0–2 read target layer 58; draft layer 3 reads target layer 59. The drafter never writes KV.
- Input: `pre_projection(concat(target_embd[t]·√5376, h))`.
- Output: logits from the draft's own `token_embd` (262144×1024), no softcap; `h_next = post_projection(·)`.
- Every draft step uses position pos0, and keys are read only up to the **committed frontier**.
- After verification, `h_pending = h_final` of the last accepted row.
- llama.cpp drafts greedily; acceptance is when the target's sample equals the draft.

## Design

**Package:** `src/models/gemma4/` → static library `gufo_gemma4`.
- It follows the `src/models/qwen38_flash_next` pattern: CPU parts are always built; HIP parts are under `ENGINE_ENABLE_HIP`; the `gufo_gemma4_reference` oracle is `EXCLUDE_FROM_ALL` and built at `-O2`.
- Docs go in `docs/models/gemma-4-31b/`, tests in `tests/models/gemma4/` (dotted names `gemma4.<name>`), tools in `tools/gemma4/`.

| File | Responsibility |
|---|---|
| `config.{hpp,cpp}` | `Config::FromGguf` (reuses the Flash-Next `Reader` helper). Reads per-layer `is_swa[]` and `head_count_kv[]`, the swa and global key length, rope dims and theta, window, softcap and eps. Rejects PLE, MoE and `shared_kv_layers≠0`. `DraftConfig::FromGguf` validates the drafter against target layers 58/59 |
| `weights.{hpp,cpp}` | `TensorRef` views. Optional `attn_v` (absent means K=V). Host copies of `layer_output_scale` and `rope_freqs`. Optional untied `output.weight`. Per-tensor type/shape checks (v1: F32/Q4_K/Q5_K/Q6_K/Q8_0). `DraftWeights`; drafter vocabulary hash must equal the target's |
| `tokenizer.{hpp,cpp}` | `Gemma4Tokenizer`: special-token matching (user-defined always, control when parse_special, longest first), ▁ normalization, newline split (a newline run that is itself a token is emitted directly), priority-queue BPE on id pairs, byte fallback, decoding (control hidden, user-defined printed) |
| `chat_template.{hpp,cpp}`, `reference/chat_template_unsloth.jinja` | Compiled renderer pinned by SHA: system/developer, user, model, tool (OpenAI forward scan), `reasoning_content` after the last user message, thinking on/off, tool declarations (`format_parameters` DSL), tool-call history (mapping or pre-serialized string), continuation after a tool response. Exposes the generation-prompt boundary for `cache_prefix_tokens` |
| `cpu_ops.{hpp,cpp}`, `reference.{hpp,cpp}` | FP32 scalar oracle: `Step(token)→logits,h_final` and `MtpStep`. Dequantizes on the fly through `src/core/quant/ggml_dequant`. Modes kFloat32 and kDecode (f16 KV rounding) |
| `engine.{hpp,cpp}` | `Model`/`Session`/`SessionSnapshot` with the same API shape as `src/models/qwen38_flash_next/engine.hpp`: `Load`, `CreateSession`, `Sync`, `Evaluate`, `DecodeStep`, `DecodeBatch`, `EvaluateBatch`, `Logits`, snapshots, `kSnapshotPayloadVersion`, `SpeculativeStats` |
| `mtp_policy.hpp`, `mtp_costs.hpp` | Draft-length controller adapted from Flash-Next's `MtpLengthController`, with a Gemma cost model (verify(width, depth) + draft_step(depth)) calibrated offline into `artifacts/mtp-costs.json`. It uses no request timings, so seeded replay holds |
| `kernels/rocm/{kernels.hpp,kernels.hip.cpp,attention.hip.cpp}` | Gemma-owned kernels (below) |
| `kernels/rocm/{device_model,executor,mtp}.{hpp,cpp}` | Weight residency, sessions, the forward pass (decode/verify/prefill), graph cache, draft/verify/accept |
| `UPSTREAM.md` | Semantic sources (llama.cpp 391fac16 files, HF revision, template hashes) and the kernel reuse policy |

**Kernel reuse:** follow the Flash-Next precedent and call the generic Qwen launchers through their headers. They are already compiled into `gufo_core`, and Gemma shapes never reach Qwen's 27B-specific branches.

- `src/models/qwen/hip/ops/gemm.hpp`:
  - `LaunchGEMV` (decode O/down, the Q5_K LM head, draft Q8_0 projections).
  - `LaunchBatchedQuantGEMMFp32` (verify and draft rows ≤8; qualified bit-identical to scalar decode).
  - `LaunchBatchedQuantGEMMPreQuantized` with the Q8_1 tile layout from `src/models/qwen/hip/kernels/prefill_quant_gemm.hpp` (all prefill projections).
- `src/models/qwen/hip/quant_ops.hpp` device functions (`QuantWarpBlockDot`, `DecodeQuantSub16`, `QuantBlockElement`) inside the Gemma fused kernels.
- `src/models/qwen/hip/ops/token.hpp`: `LaunchGPUArgmax`/`Batched`, `LaunchGPUSampling`, `LaunchGPUSpeculativeSampling`, `LaunchEmbeddingLookup`.
- Flash-Next's `mtp_sampling.hpp` (proposal/verify).
- Core pieces: `GgufReader`, `gguf_identity`, `snapshot_transfer.hpp`, `sampling.hpp`, and the Qwen27B huge-page registered weight regions (`src/models/qwen/hip/model_loader.cpp`, worth +2% AR).

A Gemma test (`gemma4.projection_ops`) pins bitwise GEMV = small-batch output for every Gemma (M, K, type) × width 1–8. It catches any Qwen kernel change that moves Gemma numbers. A kernel that needs Gemma-specific tuning is then forked into the package.

**Gemma-owned kernels.** Every decode kernel is the B=1 instance of its batched kernel, so AR and verify arithmetic are identical by construction.

1. `EmbedScaleNorm`: embedding row × `sqrtf(5376)`, then the first `attn_norm`.
2. Fused QKV multi-GEMV with per-segment types and a no-V variant for global layers. Start as separate `LaunchGEMV` calls; fuse once the bitwise test passes.
3. `QkvPostWrite`: per-head q/k norms, NEOX RoPE (per-layer θ, frequency factors, skipping identity pairs), the unweighted V norm, and f16 KV writes into the global linear cache or ring slot `pos % R`.
4. Decode/verify attention, sliding hd256 (GQA 2): keys `[max(0, p−1023), p]` from the ring; split-K chunks aligned to absolute positions; merge in fixed order; scale 1.
5. Decode/verify attention, global hd512 (GQA 8): split-K; K/V shared across the 8 query heads and all verify rows. KV traffic is 2.7 GB per token at 32K, so reuse matters.
6. Prefill attention, sliding: a fork of the Qwen `attention_wmma.hip` design with 32/16 heads, scale 1, no gate, a window lower bound per tile, and ring addressing.
7. Prefill attention, global hd512. v1 uses hipBLAS strided-batched F16 QKᵀ, a masked softmax, then P·V, split into query sub-chunks to bound S. A fused WMMA flash kernel is backlog work: at hd512, 16 rows × 512 accumulators is 256 VGPRs.
8. `PostAttnNorm`: residual plus post-norm plus `ffn_norm`, with the Q8_1 output for prefill.
9. GeGLU: gelu_tanh(g)·u through explicit non-contracting device helpers; the prefill variant quantizes to Q8_1.
10. `PostFfnScaleNorm`: residual plus post-norm, × layer scale, plus the next layer's norm; final rows go to `h_final`.
11. Softcap on logits, before sampling/logprobs. A separate kernel first, later fused into the LM head.
12. MTP helpers: draft input staging, top-64 candidate compaction, and on-device greedy acceptance.

**KV layout per session:**
- Global: token-major f16 `[ctx][4][512]` for K and V × 10 layers = 80 KiB/token, which is 10 GiB at 131K and 20 GiB at 262K. The docs recommend `--context 131072`.
- Sliding: an f16 ring `[R][16][256]` for K and V × 50 layers, with R = 1024 + prefill chunk (2048), which is 2.34 GiB.
- Why R works: no forward pass ever overwrites a slot inside a live window. Rollback after rejected drafts is therefore just moving the frontier: no recurrent state and no rollback buffers.
- Scratch is shared per executor, about 1.5 GB at chunk 2048.
- HIP graphs capture decode (B=1), verify widths 2..k+1 and the draft chain. They read a device `Control{position, frontier, R}` block so they replay at any position.

**MTP cycle** (one host sync):
1. Draft k steps at pos0. Each step reads the layer-58 ring over `[pos0−1023, pos0)` and layer 59's global KV over `[0, pos0)`.
2. Verify `[id_last, d1..dk]` with the small-batch GEMMs.
3. Accept greedy (on device) or sampled (`LaunchGPUSpeculativeSampling`, pre-drawn uniforms in a fixed RNG order).
4. Set frontier = pos0 + n + 1 and `h_pending = h_final[n]`. A stop token truncates the commit.

For sampling, measure two routes under the user's sampler and keep the faster one by completed-token rate:
- (A) greedy draft plus target-sample verification, equivalent to llama.cpp;
- (B) top-64 q with p/q rejection and residual correction, as in Flash-Next. This may raise acceptance above llama.cpp's 33–50%.

**Serving:**
- Add `Gemma4TextRunner` in `src/cli/serve/inference_backend.cpp`, modelled on `QwenFlashNextTextRunner` (~2270). `load()` gets a `"gemma4"` branch before the Qwen fallback.
- Capabilities: incremental prefill, snapshot, fork, exact incremental text, prefix reuse, and multi-token decode when MTP is on.
- Snapshots hold global rows `[0,n)`, the ring rows for `[n−1023, n)` in logical order, last logits, `h_pending` and controller state.
- The persistence identity combines the target and drafter `GgufIdentityHex`, the template SHA, the KV layout tag, context, draft limit and payload version.
- `ContinuationCache` reuses only exact checkpoints, so the ring never needs truncating. `Session::Sync` keeps a prefix only if `len − p ≤ R − 1023`.
- Trimming history still re-prefills, as llama.cpp does without `--swa-full`; faster prefill is the mitigation.
- Reuse the existing flags `--speculative mtp --mtp-model PATH -d N`. Only help text changes.

**openai_chat refactor** (Qwen and DeepSeek behavior unchanged):
- Add an `OutputSyntax {kQwen, kDsml, kGemma4}` to the runner descriptor.
- It replaces the hardcoded `<think>`/`</think>` in `ParseGeneration` (~1028) and `StreamingTextFilter` (~1273), and `kToolMarkers` (line 55).
- Add `ParseGemmaCalls`: the DSL with `<|"|>` strings, numbers, booleans, null, nested objects with bare keys, and arrays, producing OpenAI JSON.
- `inference_backend.cpp` ~2742: count reasoning tokens using the runner's reasoning-end marker instead of tokenizing `</think>`.
- `InitialOutputState` returns content when thinking is off and auto when it is on; after a tool response with thinking on, it returns reasoning.

## Milestones

Each milestone is a testable state with Conventional Commits on `feat/gemma`. M4 and M5 can overlap.

**M0 — References and fixtures (no engine code).**
- Check that `.#llama-cpp-reference` (b11069, rev 68d9053a in `.devops/nix/llama-cpp-reference.nix`) loads `gemma4` and serves `--spec-type draft-mtp` with `gemma4-assistant`. If it does not, pin a revision at or after 391fac16 as a separate reference output.
- Add `tools/gemma4/`:
  - `tokenizer_goldens.py`: token-id hashes from HF `tokenizers` (google/gemma-4-31B-it tokenizer files at a pinned revision) and `llama-tokenize`. Corpus: spaces/tabs, newline runs, CJK, emoji, invalid UTF-8, literal special tokens, a 20k-character line.
  - `template_goldens.py`: jinja2 renders under a `"gemma4"` key in `tests/fixtures/chat_template_hf_goldens.json`.
  - A `llama-perplexity --kl-divergence-base` generator over templated chat, prose and code at `-c 4096` and `-c 16384`. Output stays outside Git.
- Capture the fork baseline by running the model-bench workload against a gemma-control container. Use a local wrapper through `--reference-binary`/`--config`; it is not committed.

**M1 — CPU package (hosted-CI testable).**
- Add config, weights, tokenizer, chat template, the Gemma tool-call syntax and the CPU oracle.
- Wire `add_subdirectory(src/models/gemma4)` and `tests/models/gemma4` in the root `CMakeLists.txt`.
- Add `gemma4\.(config|tokenizer_contract|template|tool_syntax)` to `cmake/Checks.cmake`.
- *Exit:* PR checks green; tokenizer and template goldens pass; the oracle's top-1 matches llama.cpp kld rows on short prefixes, with KL recorded.

**M2 — Correct GPU AR path.**
- Build the device model, executor, ring and global KV.
- Implement kernels 1–5 and 8–11 in simple form, the prefill GEMMs through Qwen, and scalar prefill attention for both geometries.
- Add the `gufo prompt`/`chat` branches (`src/cli/prompt/prompt.cpp`, before Qwen template rendering at ~990/1204) and `RunGemma4Benchmark` (`src/cli/bench/bench.cpp`, modelled on `RunQwen38FlashNextBenchmark` ~874).
- Link `gufo_gemma4` into `gufo_llm_cli`, `gufo`, and every test target that compiles `inference_backend.cpp`.
- *Exit:* operator oracles pass; GPU vs CPU oracle passes; teacher-forced KL vs llama.cpp at 4K and 16K is within the declared limit; `gufo bench` runs.

**M3 — AR performance.**
- Fused QKV and post-norm kernels; HIP graph decode.
- Huge-page registered weights; tuned split-K decode attention.
- WMMA sliding prefill attention; fused norm→Q8_1 and GeGLU-quantize boundaries.
- Follow `.agents/skills/optimize-kernel/SKILL.md`, using `tools/prof/prof.py` and `tools/bench/gfx1151_peak.hip`.
- *Exit:* AR and prefill targets below met, with the M2 quality gates unchanged.

**M4 — HTTP serving (AR).**
- `Gemma4TextRunner`, the `OutputSyntax` refactor, snapshots and disk persistence, and the reasoning-token count.
- *Exit:* the openai_chat, session and continuation tests pass; the SDK check passes; a SillyTavern-style multi-turn replay reuses the prompt snapshot; a disk restart restores the cache.

**M5 — MTP.**
- Drafter config and weights, the draft chain, verify, acceptance, sampled routes A/B, the controller, and a graph-captured cycle. MTP must also work over HTTP.
- *Exit:* greedy MTP == AR in 100% of cases; sampled/replay tests pass; the MTP target below is met.

**M6 — Productization.**
- Batched decode at C2–C8 (per-row KV descriptors, weight reuse across requests).
- Fill `docs/models/gemma-4-31b/artifacts/bench.json` and the tables.
- Write the docs: `docs/models/gemma-4-31b/{README,BENCHMARKS,QUALITY,EXPERIMENTS}.md`, `artifacts/model-identities.json`, a row in `docs/models/README.md`, `"gemma-4-31b"` in `REQUIRED_DOC_FILES` of `tools/ci/check-docs.py`, and the MTP flag text in `docs/SERVER.md` and `docs/CLI.md`.

**M7 — Backlog.** One mechanism at a time; each is kept or deleted based on measurement, and each gets a row in EXPERIMENTS.md.
- Store only V for global layers (K = c·rope(V), because `k_norm` is constant per layer in this checkpoint). It halves global KV size and traffic, raising the AR ceiling at 131K from 7.9 to 9.6 t/s. It needs KL requalification.
- A fused WMMA hd512 flash attention kernel.
- Gemma-shape wave64 verify kernels.
- A GeGLU epilogue in the K-quant WMMA GEMM.
- Draft LM-head compaction.
- Window-only re-prefill to rebuild ring state from the global KV.
- Q4_0/QAT, untied output, fine-tune template hashes.
- Vision.

## Performance targets (vs the gemma-control fork)

| Workload | Fork | Target |
|---|---:|---:|
| pp512 / pp8192 | 348 / 307 t/s | ≥450 / ≥420 t/s |
| 33K cold prompt | 150–180 t/s (3.5 min) | ≥350 t/s (~95 s) |
| AR tg128, depth 0 | 11.40 t/s | ≥11.8 t/s (ceiling 12.8) |
| AR at 32K | to be measured | ≥10.0 t/s (ceiling 10.8) |
| MTP, depth 0, user sampler | 20.2 t/s | ≥23 t/s |
| MTP at 32K | 9.1 t/s (Melinoe MTP2) | ≥16 t/s |

## Critical files

- New: `src/models/gemma4/**`, `tests/models/gemma4/**`, `tools/gemma4/**`, `docs/models/gemma-4-31b/**`.
- Build: `CMakeLists.txt` (subdirectory and links, ~54–65, ~175–228, ~366); `cmake/Checks.cmake`; the CMakeLists of `tests/models/{qwen27b,qwen38_flash_next,deepseek_v4_flash}` (link `gufo_gemma4` where `inference_backend.cpp` is compiled).
- Serving: `src/cli/serve/inference_backend.{hpp,cpp}`, `src/cli/serve/openai_chat.cpp`, `src/cli/serve/text_model_runner.hpp` (the `OutputSyntax` field), `src/cli/serve/serve.cpp` (help text).
- CLI: `src/cli/prompt/prompt.cpp`, `src/cli/bench/bench.cpp`.
- Fixtures and CI: `tests/fixtures/chat_template_hf_goldens.json`, `tools/ci/check-docs.py`, and `.devops/nix/llama-cpp-reference.nix` if the pin must move.

## Verification

- **Per change:** run the smallest focused target, e.g. `nix develop -c cmake --build --preset gpu-test --target <t>` then `ctest --preset gpu-full -R '^gemma4\.<t>$'`. Before committing C++, run `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`. Run `nix build .#checks.x86_64-linux.pr` when the CPU contract tests change.
- **Model suites:** `tools/gemma4/check.py {fast,kernels,model,serving}` with `GUFO_GEMMA4_MODEL` and `GUFO_GEMMA4_MTP_MODEL`. A missing model exits 77, and a skip is not a pass. It covers:
  - Operator oracles: ring wrap; windows below, at and above 1024; freq-factor pairs at positions 0, 1023, 1024, 65535 and 131071; widths 1/2/5/8.
  - GPU vs CPU oracle.
  - Teacher-forced KL, top-1 and PPL delta vs llama.cpp at 4K and 16K.
  - Greedy MTP == AR on ≥20 prompts × depths 0/4K/32K × k=1..4.
  - Snapshot/restore bitwise at n = 1, 1023, 1024, 3071, 3072, 33K; disk reload; fork; cancellation during prefill and MTP.
- **End to end:** `nix build`, then `./result/bin/gufo serve llm --model $GEMMA --speculative mtp --mtp-model $MTP --context 131072`. Exercise it with `tools/serving/check-openai-sdk.py`, `check-continuation.py` and a SillyTavern-style multi-turn replay, covering thinking on/off and tool calls.
- **Benchmarks:** `tools/bench/model-bench.py --model gemma-4-31b run --target gufo|reference`. Report the fork baseline separately, with a clear label. Record results in BENCHMARKS.md and evidence in QUALITY.md.

## Risks

- **Thin AR margin at depth 0:** the fork is already at 89% of the bandwidth ceiling. The wins are prefill, depth and MTP.
- **hd512 WMMA register and LDS pressure:** the GEMM-based global prefill attention is the v1 fallback.
- **Snapshot size:** about 3.4 GB at 33K, and the default disk cache is 8 GiB. Trimmed histories still need a full re-prefill.
- **Qwen kernel coupling:** the `projection_ops` bitwise tripwire guards against it. Fast-math contraction drift is controlled with explicit-FMA helpers and bitwise B=1 vs verify tests.
- **Reference pin:** it may lack Gemma 4 or MTP support, which is settled in M0. The fork baseline is Vulkan while the repo reference is ROCm; label both.
- **Tokenizer divergences:** HF and llama.cpp may split newline runs differently. HF is primary; document any divergence.

## Progress

**2026-09-26 — M0/M1 host package.**
- The repo reference `.#llama-cpp-reference` (b11069, `68d9053a`) already
  supports `gemma4`, `gemma4-assistant` and `--spec-type draft-mtp`: no new pin.
- Tokenizer goldens (`tools/gemma4/tokenizer_goldens.py`, 20 entries): HF
  `tokenizers` (unsloth/gemma-4-31B-it@51c9f626) and llama.cpp agree on all
  19 valid-UTF-8 entries; Gufo's `Gemma4Tokenizer` matches llama.cpp on all 20.
- Template goldens (`tools/gemma4/template_goldens.py`, 19 cases rendered by
  jinja2 from the GGUF's own template): the compiled renderer is
  byte-identical on every case.
- CPU oracle vs llama.cpp (ROCm, flash attention) on a 22-token chat
  prompt: every non-matmul operator of layers 0 (sliding) and 5 (global)
  agrees to float rounding (embedding scale, all norms, K=V, proportional
  rope, attention, GeGLU). The projections differ by 0.3–1.6% relative:
  llama.cpp's integer kernels, not semantics (emulating Q8_1 activations
  does not remove it). Full-model teacher-forced result: mean KL 0.038 nats
  (max 0.34), top-1 21/22, the miss at a 0.29-probability position. Gufo's
  quality bar is to stay at or below llama.cpp's distance from the exact
  oracle.

**2026-09-26 — M2 GPU autoregressive path.**
- `gufo prompt`/`chat` run Gemma 4 on ROCm. First unoptimized decode:
  9.6 tok/s (120 tokens, greedy); load 3.4 s.
- Decode arithmetic (FP32 activations, split-K attention) vs the exact CPU
  oracle: mean KL 2.4e-7, top-1 22/22. Eight-row verification is bit-identical
  to single-token decode. Prefill (Q8_1 activations, W8A8 WMMA): mean KL
  0.0084 vs the oracle — 4.5x closer than llama.cpp (0.038).
- 320-token oracle run: Gufo matches it at every sampled position (KL ≤ 3e-5);
  llama.cpp departs at a high-entropy position in all its configurations.
- 16K in-distribution (templated model turn): KL(llama.cpp ‖ Gufo) 0.022,
  top-1 120/128, true-token NLL 3.347 (Gufo) vs 3.349 (llama.cpp). Raw text
  is off-distribution for the IT model (NLL ~9–10 nats) and magnifies noise.
- Fixed: fast-math `sinf`/`cosf` lost accuracy at large rope angles; rope now
  evaluates the float angle's cosine/sine in double.
- Host note: on Fedora 44 (glibc 2.43) Nix's hipClang links against the
  system libm through `-L/usr/lib64`; dev builds here are configured with
  `-DCMAKE_EXE_LINKER_FLAGS=-L<nix glibc>/lib` (production `nix build` is
  sandboxed and unaffected).

**2026-09-26 — M3 (first pass) and M5 MTP.**
- WMMA prefill attention (binary16 operands, FP32 accumulation): pp2048
  247 → 414 tok/s; prefill KL vs the oracle 0.0084 → 0.0039.
- Gemma decode GEMV (lane groups share activations): AR tg128 9.57 → 10.47
  tok/s. Measured DRAM read ceiling here: 239 GB/s (AR ceiling ~12.7 tok/s).
- MTP with the Unsloth drafter (on-device draft chain, verification rows
  through the batch-invariant small-batch projections): greedy MTP equals AR
  on every tested prompt. Greedy prose, 126 tokens: n=2 20.6, n=3 21.6,
  **n=4 22.3 tok/s** (acceptance 36.8%), n=5 21.4 — vs the llama.cpp fork's
  20.2 tok/s at n=4. With a drafter loaded, single-token projections use the
  shared kernels' bit-identical one-row paths so verification equals decode.

**2026-09-26 — M4 HTTP serving.**
- `gufo serve llm` loads Gemma 4 (with or without `--speculative mtp`): plain
  and streamed chat, the thinking channel as `reasoning_content`, Gemma tool
  calls as OpenAI `tool_calls` (ids `call_…`) and tool-result continuation.
- The output markup (reasoning markers, tool syntax) is per runner; Qwen and
  DeepSeek parsing is unchanged (93 CPU/server tests pass).
- Session snapshots (global rows, the live sliding rows in logical order,
  frontier hidden, logits) back the RAM and disk prompt caches. Multi-turn
  replay: 502 then 597 of 604/701 prompt tokens reused; after a restart the
  disk cache restored 502 tokens in 441 ms. `gemma4.target` checks that a
  snapshot taken after the ring wraps restores bitwise.

**2026-09-26/27 — decode and depth performance, M6 benchmarks and docs.**
- Decode GEMV split across the eight waves of a workgroup with one header
  load per super-block (all K-quant decode projections): AR 10.51 → 11.45 tok/s
  in `gufo bench`.
- Split decode attention per KV head with all grouped query heads per block
  and splits from the window start: tg at d32K 2.62 → 9.29 tok/s.
- Prefill attention tiles shared by four heads with register prefetch:
  pp2048 at d32K 205 → 262 tok/s.
- HTTP benchmarks (see BENCHMARKS.md): vs llama.cpp b11069 pp +30% (d0) to
  +99% (128K), AR tg +7–9%, MTP mixed −3% to +9%, repetitive +7% to +211%;
  vs the gemma-control fork pp +27–73%, AR tg +0–8%, MTP mixed −15% to +2%.
- Open: MTP cycle cost (5-row verification projections ~12 ms above AR,
  5-row sliding attention ~9 ms per cycle past the window), batched decode
  for C>1, and greedy MTP equality with an AR-only server (needs a
  verification kernel with the AR GEMV's summation order at shared-kernel
  speed).

**2026-09-27 — MTP draft argmax.**

- Reused the idle draft FFN gate scratch for the shared two-pass GPU argmax.
  Its median latency dropped from 272.3 to 9.9 + 1.9 µs; the real-model
  `gemma4.target` test passed. A focused release HTTP prose retest through 64K
  gained 0.5–3.2% over the prior Gufo rows without changing acceptance.
  The fork remains ahead at six of seven measured depths, so verification
  projection and attention cost are still open.

**2026-09-27 — MTP cycle cost: row-shared split attention.**

- Replaced the per-row split decode/verification attention with one wave per
  (query head, chunk) serving up to five rows from registers (online softmax
  per 32-key tile, DPP lane exchanges, parallel split merge), then staged the
  hd512 K/V tiles once per block in shared memory. Verification still equals
  decode bit for bit; decode mean KL vs the oracle 9.8e-7 → 5.5e-7.
- d32K five-row verify attention: global 4.79 → 1.82 ms, sliding 198 → 65 µs;
  one-row global decode 1.56 → 1.36 ms.
- HTTP prose MTP now leads llama.cpp by 10–75% and the fork at six of seven
  measured depths (−3.3% at 4K, where the fork's text accepts 1.91 drafts per
  cycle vs 1.67; Gufo's cycle is still shorter). AR tg +10–12% vs llama.cpp.
- Rejected: LDS-staged values in the old kernel, a bit-exact multi-row Gemma
  GEMV for verification (18% slower than the shared small-batch kernel).
- Open: five-row verification projections (~92 ms per cycle vs a ~79 ms
  bandwidth floor), drafter head and global attention (~15 ms per cycle at
  32K), batched decode for C>1.
