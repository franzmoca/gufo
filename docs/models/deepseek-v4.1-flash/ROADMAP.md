# DeepSeek V4.1 Flash Roadmap

Goal: find out whether DeepSeek-V4.1-Flash-1017 is usable on one Strix Halo
(gfx1151, 124 GiB GTT, NVMe for streaming) with the
[ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF):

| Artifact | Size | Placement |
|---|---|---|
| `IQ1_M` backbone, 4 shards | 112.4 GiB | resident |
| `IQ2_XXS` backbone, 4 shards | 128.7 GiB | resident dense weights, experts streamed from SSD |
| `Engram-IQ4_NL.gguf` (`deepseek41-engram`) | 103.0 GiB, 144 B rows | always read from disk |

The engine is `src/models/deepseek_v41_flash/`, a fork of the V4 Flash engine;
V4 is unchanged. `tools/ds41/` holds an independent CPU/torch oracle that
ports the official `inference/model.py` and `inference/engram.py` and reads the
same GGUF.

## Status

| Milestone | State |
|---|---|
| M0 Oracle and loader | done |
| M1 Correct AR decode on IQ1_M | done |
| M2 Resident performance | done |
| M4 Component cache and serving | done |
| M5 DSpark, prompt lookup and approximations | in progress: DSpark done |
| M6 Vision | planned |
| M3 SSD expert streaming (IQ2_XXS) | planned, after M6 |
| M7 Higher-precision Engram tables | planned |

Order: M4, M5, M6, M3, then M7. M4-M6 build on the resident IQ1_M model;
M3 then measures streamed IQ2_XXS experts against it, with DSpark and the
Engram reads already in place, since both compete with streaming (DSpark's
multi-token verification for expert reuse, Engram for the SSD). M7 measures
Engram table formats on the final configuration.

## Memory budget

MemTotal is 124.4 GiB: 128 GiB minus the 1 GiB UMA carve-out and about
2.6 GiB of firmware and kernel reservations. `amdgpu.gttsize=126976`
(124.0 GiB) already covers it, so a larger GTT gains nothing; shrinking the
carve-out to 512 MB is the only remaining knob (+0.5 GiB).

Measured IQ1_M load with `ds41_load_probe --upload`: 18 s, 112.27 GiB GTT
(the token embedding stays in the mapped file), 9.3 GiB `MemAvailable`
afterwards. A session at the native 1M context adds 2.3 GiB: 0.9 GiB of FP4
compressed state and 1.4 GiB of batch scratch shared by all sessions; a
2048-token prefill peaks 0.3 GiB higher for the MMQ pool. The upload streams from the file descriptors
and never doubles the weights in page cache; the Engram file must not be
resident in page cache at the same time.

## What V4.1 changes versus V4 Flash

| Area | V4 Flash | V4.1 Flash |
|---|---|---|
| Shape | 43 layers, 4096 hidden, 256 experts, ffn 2048, q_lora 1024, 64 indexer heads | 40 layers (+3 DSpark layers not in the ISTA builds), 5120 hidden, 384 experts top-6 + 1 shared, ffn 2304, q_lora 1280, 32 indexer heads, RMS eps 1e-20 |
| Routing | 3 hash layers | none; sqrtsoftplus + bias top-6, normalized, x1.5 |
| Layout | uniform | YoCo: layers 0-19 encoder, 20-39 decoder |
| Compression | ratio 4 (overlapping) / 128 | ratio 0 = SWA only (0-1); 2 = softmax-gated pool (2-19); 1 = RMSNorm(wkv(x)) (20-39). RoPE base 160000 + YaRN on ratio >= 1 |
| KV/index sharing | per layer | CSA2: layers 2, 8, 14, 20 own compressor and index K; 24, 28, 32, 36 rescore layer 20's index K with their own Q; all others reuse the latest top-512. Index K is projected from the main latent (512 -> 128) |
| Candidate blocks | none | layer 20 takes the block max over 8-row blocks, keeps the top 2048 blocks (newest pinned) and masks the rescoring layers; a no-op up to 16K context |
| mHC | mixes used on the same layer | shifted: attention uses the previous FFN's `pre`; layer 0 uses a one-hot `pre`; no `output_hc` |
| Engram | none | layers 1 and 14, before the block: 24 rows per token (2- to 4-grams x 8 heads over 99,092 compressed ids), `wkv` 6144 -> 5 x 5120, per-copy sigmoid gate, `h_c += gate_c * v` |
| Template | `<｜DSML｜tool_calls>` | DSML tags with a leading space, `Reasoning Effort: N` (1-100; high = 75 default), mid-conversation `<｜System｜>` turns |

State is small: about 890 B/token of compressed KV and index K plus a fixed
128-row SWA ring per layer at the reference precision; stored as FP4 codes
and scales it is about 0.9 GiB at 1M tokens.

GGUF types: experts are IQ1_S/IQ1_M/IQ2_XXS/IQ2_S per layer (gate and up
share a type; down may differ); dense projections Q8_0/Q5_K; router,
compressor, indexer and Engram q/k BF16; hc and norms F32; `token_embd` and
`output` Q5_K; Engram index tensors I32/I64.

## Performance expectations

IQ1_M reads about 6.9 GB per decode token: dense weights about 4.7 GB, output
0.45 GB, routed experts about 1.7 GB. At a practical 135-200 GB/s that gives a
ceiling of 20-29 tok/s; 12-18 tok/s is the resident target. Published
single-Halo results with SSD streaming reach about 10 tok/s decode. Streaming
needs about a 99 % per-use expert hit rate to keep about 90 % of resident
speed.

## Milestones

### M0 Oracle and loader (done)

- `tools/ds41/reference.py`: CPU/torch oracle with official KV quantization
  (FP8 SWA, FP4 compressed and indexer), chunked sparse attention with sinks,
  candidate blocks and Engram. It dumps logits, routes and per-layer states.
  `verify-engram` rebuilds primes, offsets, multipliers and the compressed
  token map and matches the GGUF exactly.
- `tools/ds41/quants.py`: IQ1_S, IQ1_M, IQ2_XXS, IQ2_S and IQ4_NL dequant,
  bit-exact with the llama.cpp reference.
- Oracle results on IQ1_M: greedy "The capital of France is Paris. The
  capital of Germany is" continues " Berlin. The capital of Italy is Rome".
  A 3K-token teacher-forced run on `docs/KV-CACHE.md` gives perplexity 6.31,
  top-1 58 %. A shifted-context control on the same tokens confirms the
  long-context run is better in every band.
- Loader: split GGUF mapped as one range, shard-aware uploads, strict
  validation of all `deepseek41.*` metadata and all 1008 tensors, Engram file
  opened by source hash and kept on disk. Both IQ1_M and IQ2_XXS validate.
- Tokenizer: identical to the official `tokenizer.json` on about 19k tokens of
  mixed text, code, scripts and special tokens.

### M1 Correct AR decode on IQ1_M (done)

- Single-token decoder (`runtime/v41_decode.cpp`) over plain kernels
  (`kernels/rocm/ds41_v41.hip.cpp`) and typed MMVQ entries for every expert
  and dense type in the ISTA builds. It covers CSA2 own/rescore/reuse modes,
  ratio-2 and ratio-1 compressors, index K from the latent, candidate blocks,
  shifted mHC, sqrtsoftplus routing and Engram (CPU hashing, synchronous
  `pread` of 24 rows per layer). All experts are resident; model spans are
  sized to the largest tensor so no expert stack is split.
- Sessions: prefix reuse and snapshots of the full decoder state.
- Chat template: official V4.1 encoding with numeric reasoning effort,
  mid-conversation system turns and V4.1 DSML tags, pinned to the GGUF
  template and the official encoder by SHA-256.
- Serving: `deepseek41` in `gufo serve` with the V4 Flash runner, a V4.1
  state ABI and cache identity, DSML grammar (`ToolFormat::kDeepSeek41`) and
  output parsing, and V4 Flash sampling defaults (temperature 1.0, top_p
  0.95), as in deepseek-recipe. The core GGUF reader now accepts the IQ1_S,
  IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS and I64 storage types.
- Results in [QUALITY.md](QUALITY.md): 3K-token perplexity 6.306 against the
  oracle's 6.309; 12/12 template goldens; working chat, thinking and tool
  loops through `gufo serve`. Decode 12-14 tok/s at short context.
- Moved to M2: `gufo prompt` and `gufo bench` support, and candidate-block
  parity beyond 16K tokens.

### M2 Resident performance

Open performance items are tracked in [BACKLOG.md](BACKLOG.md).

- Done: `deepseek41` in `gufo prompt` and `gufo bench`
  ([BENCHMARKS.md](BENCHMARKS.md)); candidate-block parity beyond 16K
  tokens ([QUALITY.md](QUALITY.md)).
- Done: batched layer-major prefill on the FP16 matrix cores (routed experts,
  dense projections, attention), fused gate/up + SwiGLU, the fused batched
  mHC state pass and `--prefill-batch`. pp2048 703 tok/s and tg64 23.9 tok/s
  at depth 0, 372 and 21.0 at 131K ([BENCHMARKS.md](BENCHMARKS.md)).
- Done: Engram reads start with the batch on parallel `pread` threads and
  overlap the GPU work. Measured waits are 0.04 ms per prefill batch and
  0.33 ms per decode token, so sorted reads, `O_DIRECT` and a row cache are
  deferred ([BACKLOG.md](BACKLOG.md)).
- Done: per-kernel prefill and decode time breakdowns
  ([BACKLOG.md](BACKLOG.md)) and ceilings from `tools/bench/gfx1151_peak.hip`
  ([EXPERIMENTS.md](EXPERIMENTS.md)).
- Done: functional suites and the Pi agent replay on the M2 build
  ([QUALITY.md](QUALITY.md)); Devlab plans cover the model.

### M3 SSD expert streaming (IQ2_XXS)

- Expert store v2: a slot pool sized from free memory, decayed-use eviction,
  scan-resistant admission during prefill, promotion of experts routed by the
  last prompt rows, and a seed hot list from a routing histogram.
- Reads sorted by offset and merged; no speculative next-layer prefetch during
  decode; prefill may prefetch the next layer's selected experts.
- Hit rate, bytes per token and stall time in `diagnose` and logs.
- Compare resident IQ1_M with streamed IQ2_XXS on quality and speed.

### M4 Component cache and serving

- `src/cache/adapter.hpp` adapter: compressed KV and index K of each source
  layer as append rows (ratio-2 layers add a row every 2 tokens); the
  128-row SWA rings, compressor frontiers and logits as private state; the
  Engram n-gram history is rebuilt from the tokens on restore.
- Disk cache and prefix reuse as for Gemma 4.

### M5 DSpark, prompt lookup and approximations

- Start with a memory budget. With one session at 1M context about 9.5 GiB
  of host memory is free; the drafter, the M4 caches and the M6 projector
  come out of it.
- DSpark drafter (done): `gufo serve --dspark-model` loads the unsloth
  (`dflash`) or Lucebox (`deepseek41-dflash-draft`) layout, Q8_0, Q2_K,
  Q4_K, MXFP4 and F16/BF16 tensors. Drafts of up to 5 tokens are verified
  in one batch with decode numerics and the request's sampler, so output
  is unchanged; the draft width follows measured acceptance and cycle
  times. Greedy decode +21% on a document prompt, +61-67% on a coding
  request ([BENCHMARKS.md](BENCHMARKS.md#dspark-speculative-decoding)).
  The drafter's rings ride in the session snapshot.
- Memory: Lucebox Q2_K/Q4_K takes 4.71 GiB and leaves 4.7 GiB of host
  memory after load; unsloth Q8_0 (7.42 GiB) leaves 2.1 GiB for similar
  speed, so Q2_K/Q4_K is the recommended drafter. Not used:
  [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
  BF16 (7.92 GiB), Lucebox `-MXFP4-Q8.gguf` (7.42 GiB) and the
  [peasantsmith](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)
  `.bin` drafters (not GGUF).
- Drafter rings as component-cache private state, so sessions with DSpark
  use the component cache instead of whole-state snapshots.
- Prompt lookup (`src/core/speculative/prompt_lookup.hpp`) as for Gemma 4.
  It shares DSpark's multi-token verification and the rollback of rejected
  tokens (SWA rings, compressor frontiers, rows), and its state joins the
  component cache metadata.
- Evaluate decoder SWA bounded replay (layers 20-39 over only the last 128
  prompt tokens), which roughly halves prefill; adopt only if quality holds.

### M6 Vision

- Vision tower from
  [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
  `mmproj-BF16.gguf` (0.91 GiB; `mmproj-F16.gguf` 0.90 GiB). The official
  config has 32 layers, width 1024, patch 14, a 3x downsample and at most
  1,024 image tokens per image (`<｜deepseek_image｜>`, token 129264).
- Interleaved image blocks in the chat template, checked against the official
  `encoding.py` image goldens; the image functional suites for this model.
- Component cache for image prompts: the image identity in the cache
  metadata, as for Qwen3.8-Flash-Next, so a restored prefix only serves
  requests with the same images; the image cache suites (`image-count` RAM
  and disk replay).

### M7 Higher-precision Engram tables

- Measure the speed and quality of Engram tables above IQ4_NL (144 B rows).
  [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
  `engram/` offers Q5_K (125.89 GiB), Q6_K (150.21 GiB) and MXFP8
  (188.83 GiB), in its own split-file format (`*.gguf_file`, `set_split.py`).
  Full BF16 tables (512 B rows, about 366 GiB) would come from the official
  checkpoint.
- Quality: teacher-forced perplexity and agreement against the oracle at 3K and
  18.5K tokens, per table format. Speed: prefill and decode at the card
  depths, and the Engram read and wait times (BACKLOG item 6), because rows
  grow up to 3.6 times.

## Verification

- Oracle parity per layer and per token, tolerances recorded per quant.
- `ctest --preset gpu-full -R '^ds41'` for the model tests; the shared format
  check.
- Functional chat, tool, long-context and cache suites for this model only,
  and `tests/functional/pi_agent.py` once tool calls work.
- `gufo bench` with matched settings; streaming hit rate; peak memory at the
  largest configured context.
