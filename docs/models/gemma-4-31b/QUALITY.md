# Gemma 4 31B quality

**Decode matches the scalar reference; greedy MTP matches autoregressive
output.** Unsloth UD-Q4_K_XL target and Q8_0 `gemma4-assistant` drafter;
[identities](artifacts/model-identities.json). The arbiter is Gufo's scalar
CPU implementation of the llama.cpp `gemma4` graph over the same GGUF
(FP32 activations, binary16 KV, `src/models/gemma4/reference.cpp`); these are
not unquantized-model or GGUF-conversion checks. Measured September 26, 2026.

| Check | Result |
| --- | --- |
| GPU decode (FP32 activations, split-K attention) vs scalar reference, 22-token chat prompt | Mean KL 9.7e-7 (limit 1e-5), top-1 22/22 |
| Eight-row verification vs scalar reference | Mean KL 1.5e-6, top-1 22/22; verification rows reproduce single-token decode bit for bit (projection widths 2–16, attention rows ≤16) |
| GPU prefill (Q8_1 activations, binary16 WMMA attention) vs scalar reference | Mean KL 0.0055 (limit 0.015), max 0.048, top-1 22/22; llama.cpp b11069 on the same prompt: 0.038, top-1 21/22 |
| Bulk prefill vs exact rows, 1542-token conversation past the window and ring (`tests/models/gemma4/fixtures/long_conversation.txt`) | Mean KL 0.056 (limit 0.15), top-1 1444/1542; 0.056–0.076 across rounding-only kernel changes |
| Bulk prefill vs exact rows, 1353-token repetitive prompt (reported, not a gate) | Mean KL 1.64, top-1 1016/1353; llama.cpp prefill vs the same rows 2.93, 920/1353. Many near-tied predictions make this prompt ill-conditioned for any binary16/Q8_1 prefill: rounding-only changes moved it between 1.33 and 1.64, so it no longer gates changes (replaced by the conversation above on 2026-09-27; its former limit was 1.6). The exact rows match the reference (KL 3.5e-6 over the first 96 tokens). |
| Long context vs llama.cpp, 16K templated turn (15.3K-token prose prompt, 640-token model reply) | KL(llama.cpp ‖ Gufo) 0.0092, top-1 628/641, true-token NLL 0.097 (llama.cpp 0.112); with the full global K cache: 0.0093, 628/641, 0.097 |
| Greedy MTP vs single-token decode with the drafter loaded, three prompts × 64 tokens, four drafts | Identical token IDs |
| Greedy MTP vs an AR-only server | Not equal in general: AR-only decode uses the faster split-K GEMV, whose FP32 summation order differs from the verification kernels, so long greedy completions can diverge (the pp2048 prose completion does). Both are FP32 decodes within the KL limits above. |
| Batched decoding | Up to eight sessions share a forward; with the drafter loaded each row reproduces its own session's decode bit for bit (`gemma4.target`: three sessions, plain and speculative greedy steps equal AR). Without the drafter, batches use the verification kernels while one session uses the faster AR GEMV, so a greedy completion can depend on concurrency (the multi-user AR table matched its C1 outputs). Multi-user MTP tables check every completion against Gufo's own MTP C1 (`exactness_reference: self` in bench.json), since greedy MTP does not equal an AR-only server |
| Session state | Prefix extension, rewind and snapshot restore after a sliding-ring wrap (typed and serialized) continue bit for bit; truncated snapshots are rejected |
| Tokenizer | Hugging Face `tokenizers` and llama.cpp agree on the 19 valid-UTF-8 entries of a 20-entry corpus (whitespace/newline runs, CJK, emoji, special tokens, 20k-character line); Gufo matches llama.cpp on all 20 |
| Chat template | Compiled renderer byte-identical to Jinja2 on 19 cases (system, thinking on/off, tools, tool history, reasoning history) |
| Global-layer keys | K and V share one projection, so the cache keeps V and only the 64 rotated rope pairs of K; the other key dims are V times `k_norm`, folded into the query. Attention with these keys matches FP64 like stored keys (split ≤5e-6, WMMA ≤0.007) |
| Kernels | Attention vs FP64 on window, ring, key-limit and 32K shapes (split ≤5e-6, WMMA ≤0.007 absolute), repeated launches bit-identical; every decode projection within 2e-6 of an FP64 dot relative to Σ\|wx\| |
| Serving | Streaming, `reasoning_content`, Gemma tool calls and tool-result turns, multi-turn prompt reuse and disk-cache restore across a restart |
| Vision encoder | GPU vs the scalar FP64-accumulating reference (`vision::Reference`): embedding relative RMS 1.6% at 260 soft tokens, 2.6% at 1,107 (BF16 projection inputs; the patch stage matches to 5e-7). llama.cpp's own encoder is 5.9% off at 260. `gemma4.vision_encoder` gates every stage |
| Image prompts | Teacher-forced chart description, text model fed Gufo's encoder vs fed the reference's embeddings: mean KL 0.0021 (260 tokens) and 0.0091 (1,107; top-1 99/101). llama.cpp with `clip.use_gelu` measured the same way: 0.0149 and 0.078. Stock llama.cpp runs the tower with GELU-quick (its converter omits `clip.use_gelu`) and lands at 0.224 |
| Image sessions | `gemma4.vision_session`: a red and a blue image behind identical tokens give different answers (KL 32); changing, rewinding, extending, snapshot-restoring and chunk-splitting image prompts all match fresh sessions. Over HTTP with `--cache-disk`, a follow-up after a restart restored 272 of 316 prompt tokens; the same conversation with a colour-inverted copy of the image (identical tokens) restored none |

Greedy MTP drafts the drafter's argmax and keeps it while the target's own
choice agrees. Sampled MTP samples each draft from the drafter's top-64
logits under the request's sampler (q) and accepts it with probability
min(1, p/q) against the target's filtered distribution p, drawing a rejected
position from the residual max(0, p − q); emitted tokens follow the target
distribution exactly (the rule is shared with Flash-Next and unit-tested
there). How many drafts a cycle verifies depends on the drafter's outputs,
earlier verification outcomes and a fixed cost table, never on timings, so
greedy equality holds under every `--draft-policy`. With `--draft-calibration
request`, a seed replays the same tokens (`gemma4.target`), also beside other
sessions (a sampled cycle then prices its drafts as if alone), though they consume
different random draws than an AR run with the same seed; with the default
shared calibration, what earlier requests taught the policy changes later
draft counts, so a repeated seed draws a different, equally distributed
sample. Prompt-lookup
copies are verified like drafts (sampled: as point-mass proposals under the
same rule); `gemma4.target` checks greedy equality and seeded replay on a
prompt that copies a paragraph, with accepted copies.

## Reproduce

```sh
export GUFO_GEMMA4_MODEL=/path/to/gemma-4-31B-it-UD-Q4_K_XL.gguf
export GUFO_GEMMA4_MTP_MODEL=/path/to/mtp-gemma-4-31B-it-Q8_0.gguf
nix develop -c cmake --build --preset gpu-test --target gemma4_target_test \
  gemma4_attention_ops_test gemma4_projection_ops_test gemma4_kernel_ops_test
nix develop -c ctest --preset gpu-full -R '^gemma4\.' --output-on-failure
```

`gemma4.target` loads the model twice (without and with the drafter) and
takes about a minute; without the environment variables it exits 77, which
is a skip, not a pass. For teacher-forced comparisons with llama.cpp:
`tools/gemma4/build_llama_logits.sh`, then `gemma4_gpu_probe --tokens T.i32
--logits-out G.g4lg` and `llama_logits --tokens T.i32 --output L.g4lg`, and
`python3 tools/gemma4/compare_logits.py L.g4lg G.g4lg`.
Image prompts: `gemma4_vision_generate_probe --image FILE --out-dir D`
(`--image-tokens`, `--continuation`, `--embeddings`) and `llama_vision
--image D/image0.rgb W H --tokens D/tokens.i32 --logits-out L.g4lg`, whose
`--mmproj` should be a copy of the sidecar with `clip.use_gelu = true`
(see [VISION.md](VISION.md)); `gemma4_vision_probe` compares encoders.

## Benchmark method

September 26–27, 2026; one warmed sample per point, greedy, thinking off. MTP
drafts up to seven tokens under Gufo's confidence-based length control
(`--draft-tokens 7`) and up to four on llama.cpp and the fork. Single-user uses pp2048/tg128 after a cached
prefix of the stated depth. Loading: cold files (page cache dropped with
`echo 3 > /proc/sys/vm/drop_caches`), C1, MTP, capacity 262144. Memory: C1, AR,
capacity 133121, peak device-global HIP allocation. The llama.cpp reference is
`b11069` (ROCm) from `flake.nix`. The gemma-control fork (halo-box/strix-llama.cpp
`8c1c282ec`, Vulkan RADV, `GGML_VK_MMV_NO_SPLIT=1 -b 2048 -ub 512`) was
measured with the same driver through a local container wrapper; its results
are in [artifacts/fork](artifacts/fork). Commands, counts and server flags are
recorded per row in the artifacts.
