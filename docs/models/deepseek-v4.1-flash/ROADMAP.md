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
| M2 Resident performance | next |
| M3 SSD expert streaming (IQ2_XXS) | planned |
| M4 Component cache and serving | planned |
| M5 DSpark and approximations | planned |

## Memory budget

MemTotal is 124.4 GiB: 128 GiB minus the 1 GiB UMA carve-out and about
2.6 GiB of firmware and kernel reservations. `amdgpu.gttsize=126976`
(124.0 GiB) already covers it, so a larger GTT gains nothing; shrinking the
carve-out to 512 MB is the only remaining knob (+0.5 GiB).

Measured IQ1_M load with `ds41_load_probe --upload`: 21 s, 112.69 GiB GTT,
8.57 GiB `MemAvailable` afterwards. KV (about 0.5 GiB at 128K) and scratch
(2-4 GiB) fit in what remains. The upload streams from the file descriptors
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

State is small: about 890 B/token of compressed KV plus a fixed 128-row SWA
ring per layer, so a 1M-token context costs about 4 GiB.

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

- `deepseek41` in `gufo prompt` and `gufo bench`; candidate-block parity
  beyond 16K tokens.
- Batched layer-major prefill; fused gate/up + SwiGLU for IQ1/IQ2 experts;
  LDS padding and bank-conflict passes.
- Engram v2: start row reads as soon as token ids are known (layer 1 overlaps
  embedding and block 0, layer 14 overlaps blocks 1-13); sorted, deduplicated
  reads per prefill chunk through `pread`/io_uring workers with `O_DIRECT`;
  a small pinned row cache. OS-specific I/O goes in a `src/core/platform/` leaf.
- Profile with `tools/prof/prof.py`; check ceilings with
  `tools/bench/gfx1151_peak.hip`.
- `BENCHMARKS.md`: prompt and generation speed at depths 0/8K/32K/131K.

### M3 SSD expert streaming (IQ2_XXS)

- Expert store v2: a slot pool sized from free memory, decayed-use eviction,
  scan-resistant admission during prefill, promotion of experts routed by the
  last prompt rows, and a seed hot list from a routing histogram.
- Reads sorted by offset and merged; no speculative next-layer prefetch during
  decode; prefill may prefetch the next layer's selected experts.
- Hit rate, bytes per token and stall time in `diagnose` and logs.
- Compare resident IQ1_M with streamed IQ2_XXS on quality and speed.

### M4 Component cache and serving

- `src/cache/adapter.hpp` adapter: SWA rings as window rows (128); compressed
  KV and index K of each source layer as append rows (ratio-2 layers add a
  row every 2 tokens); compressor frontiers, logits and the Engram n-gram tail
  as private state. State ABI `deepseek-v4.1-flash-gfx1151-state-v1`.
- Disk cache, prefix reuse and prompt lookup as for Gemma 4.

### M5 DSpark and approximations

- A drafter GGUF from the official `mtp.*` blocks (3 SWA blocks, 128-expert
  MoE, Markov head, confidence head) with block-of-5 non-causal drafts.
- Evaluate decoder SWA bounded replay (layers 20-39 over only the last 128
  prompt tokens), which roughly halves prefill; adopt only if quality holds.

## Verification

- Oracle parity per layer and per token, tolerances recorded per quant.
- `ctest --preset gpu-full -R '^ds41'` for the model tests; the shared format
  check.
- Functional chat, tool, long-context and cache suites for this model only,
  and `tests/functional/pi_agent.py` once tool calls work.
- `gufo bench` with matched settings; streaming hit rate; peak memory at the
  largest configured context.
