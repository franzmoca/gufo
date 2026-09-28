# Gemma 4 31B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target
and its `gemma4-assistant` Q8_0 MTP drafter. Gufo drafts up to seven tokens
under its confidence-based length control; llama.cpp drafts up to four.
HTTP, greedy, thinking off. llama.cpp is the repository's pinned `b11069`
(ROCm) reference for AR and MTP.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 427.05 | 300.91 | +41.9% | 10.73 | 9.80 | +9.5% |
| 4,096 | 390.56 | 264.62 | +47.6% | 10.60 | 9.61 | +10.3% |
| 8,192 | 368.90 | 239.48 | +54.0% | 10.46 | 9.43 | +10.9% |
| 12,288 | 345.49 | 215.96 | +60.0% | 10.37 | 9.26 | +12.0% |
| 16,384 | 343.90 | 202.00 | +70.2% | 10.24 | 9.11 | +12.4% |
| 32,768 | 294.07 | 154.12 | +90.8% | 9.81 | 8.51 | +15.3% |
| 65,536 | 225.69 | 104.28 | +116.4% | 9.07 | 7.53 | +20.5% |
| 131,072 | 156.09 | 64.56 | +141.8% | 7.84 | 6.12 | +28.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-mtp -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 426.37 | 294.20 | +44.9% | 23.15 | 19.08 | +21.3% | 44.05 | 31.11 | +41.6% |
| 4,096 | 396.50 | 253.33 | +56.5% | 23.05 | 20.32 | +13.4% | 46.54 | 32.28 | +44.2% |
| 8,192 | 367.71 | 231.63 | +58.7% | 21.83 | 18.58 | +17.5% | 41.28 | 28.74 | +43.6% |
| 12,288 | 346.46 | 211.21 | +64.0% | 21.50 | 18.14 | +18.5% | 42.83 | 28.37 | +51.0% |
| 16,384 | 343.63 | 195.22 | +76.0% | 21.63 | 15.95 | +35.6% | 40.88 | 27.19 | +50.3% |
| 32,768 | 295.05 | 150.59 | +95.9% | 20.92 | 14.89 | +40.5% | 36.47 | 21.08 | +73.0% |
| 65,536 | 225.88 | 102.73 | +119.9% | 15.73 | 10.56 | +49.0% | 31.82 | 14.85 | +114.3% |
| 131,072 | 157.46 | 63.79 | +146.8% | 12.61 | 6.92 | +82.2% | 23.64 | 4.09 | +478.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Single user, sampled MTP

The sampler of a typical chat front end: temperature 1, top-k 64, top-p 0.95,
repeat penalty 1.05; a story-writing turn after the cached prefix (the prefix
turn itself is greedy). Mean of three requests with seeds 1–3; sampled text
differs between engines and runs, so acceptance varies more than in the
greedy tables.

<!-- bench:single-mtp-sampled -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 418.29 ± 3.52 | 293.21 ± 0.96 | +42.7% | 19.78 ± 0.80 | 18.12 ± 1.97 | +9.2% |
| 4,096 | 389.09 ± 3.03 | 256.40 ± 2.11 | +51.8% | 20.47 ± 1.31 | 16.51 ± 0.64 | +24.0% |
| 16,384 | 335.90 ± 3.80 | 195.75 ± 1.13 | +71.6% | 18.26 ± 0.75 | 14.03 ± 1.35 | +30.1% |
| 32,768 | 292.59 ± 4.28 | 150.21 ± 0.25 | +94.8% | 16.99 ± 1.29 | 10.62 ± 0.56 | +60.0% |
| 65,536 | 225.93 ± 1.08 | 102.75 ± 0.08 | +119.9% | 15.11 ± 1.29 | 9.37 ± 0.88 | +61.3% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
Gufo decodes up to eight sessions in one forward: row-wise work runs once for
all of them, attention per session. Every session reads its own sliding-window
cache (16 MB per layer), about a fifth of an eight-user step.

<!-- bench:multi-ar -->
| Gemma 4 31B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 10.71 | 9.79 | +9.4% |
| 2 | 19.44 | 17.54 | +10.8% |
| 4 | 32.73 | 28.83 | +13.5% |
| 6 | 43.81 | 34.03 | +28.7% |
| 8 | 47.01 | 35.08 | +34.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table. Gufo verifies all
sessions' drafts in one forward of at most 16 rows (16 / users − 1 drafts per
session), which keeps each row's arithmetic equal to its own session's decode;
llama.cpp verifies up to 4 drafts per user with Q8_1-activation matrix kernels,
which scale further with rows. Gufo's lead from AR does not carry past two
users.

<!-- bench:multi-mtp -->
| Gemma 4 31B Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23.27 | 20.73 | +12.3% | 59.68 | 30.93 | +93.0% |
| 2 | 30.80 | 32.25 | -4.5% | 68.28 | 49.44 | +38.1% |
| 4 | 39.00 | 48.28 | -19.2% | 63.44 | 75.05 | -15.5% |
| 6 | 48.04 | 58.11 | -17.3% | 53.60 | 99.65 | -46.2% |
| 8 | 50.13 | 57.05 | -12.1% | 55.58 | 87.82 | -36.7% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)


## gemma-control fork (single user)

The previous production setup: halo-box/strix-llama.cpp `8c1c282ec` on Vulkan
RADV with `GGML_VK_MMV_NO_SPLIT=1 -b 2048 -ub 512`, same GGUF files, same
driver workloads and four draft tokens (Gufo up to seven). Hand-rendered from
[artifacts/fork](artifacts/fork). At 128K with MTP its Vulkan queue timed out
(`Fence fallback timer expired on ring comp_1.1.0`); its repetitive MTP
workload was not run.

Greedy MTP texts differ between the engines, so accepted drafts per cycle
differ per depth; Gufo drafts up to seven tokens under its confidence-based
length control, the fork a fixed four. Gufo's cycle (draft plus verify) is
shorter at every measured depth: 118 vs 120 ms at d0, 121 vs 126 ms at 4K,
136 vs 165 ms at 32K and 160 vs 211 ms at 64K.

| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 427.05 | 293.92 | +45.3% | 10.73 | 10.53 | +1.9% |
| 4,096 | 390.56 | 259.83 | +50.3% | 10.60 | 10.29 | +3.0% |
| 8,192 | 368.90 | 243.79 | +51.3% | 10.46 | 10.01 | +4.5% |
| 12,288 | 345.49 | 220.65 | +56.6% | 10.37 | 9.87 | +5.1% |
| 16,384 | 343.90 | 211.76 | +62.4% | 10.24 | 9.58 | +6.9% |
| 32,768 | 294.07 | 164.54 | +78.7% | 9.81 | 8.93 | +9.9% |
| 65,536 | 225.69 | 117.97 | +91.3% | 9.07 | 7.74 | +17.2% |
| 131,072 | 156.09 | 74.00 | +110.9% | 7.84 | 6.10 | +28.5% |

---

| Gemma 4 31B Q4 MTP mixed<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain | Gufo / fork accepted per cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 418.67 | 286.71 | +46.0% | 23.15 | 21.71 | +6.6% | 1.72 / 1.61 |
| 4,096 | 391.78 | 253.75 | +54.4% | 23.05 | 23.17 | -0.5% | 1.78 / 1.91 |
| 8,192 | 366.59 | 234.69 | +56.2% | 21.83 | 20.84 | +4.8% | 1.72 / 1.78 |
| 12,288 | 346.24 | 216.64 | +59.8% | 21.50 | 17.49 | +22.9% | 1.61 / 1.37 |
| 16,384 | 343.63 | 201.12 | +70.9% | 21.63 | 19.42 | +11.4% | 1.72 / 1.72 |
| 32,768 | 294.95 | 159.66 | +84.7% | 20.92 | 16.47 | +27.0% | 1.84 / 1.72 |
| 65,536 | 225.88 | 114.11 | +97.9% | 15.73 | 12.67 | +24.2% | 1.51 / 1.67 |
| 131,072 | 157.46 | N/A | N/A | 12.61 | N/A | N/A | 1.56 / N/A |

## Loading time

C1, capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 31B Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 4.41 | 6.58 | +49.2% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 133121 tokens, AR. Both engines allocate KV for the whole capacity,
so the footprint barely grows with the prefix. The counter is device total
minus free, which on this unified-memory APU tracks system-wide use; both
engines were measured on September 27 from the same 3.0 GiB idle baseline.
Gufo's global layers store V and only the rotated key dims (50 KiB per
token instead of 80).

<!-- bench:memory -->
| Gemma 4 31B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 32.87 | 35.47 | +7.9% |
| 16K prefix, pp4096 + tg128 | 34.26 | 37.64 | +9.9% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
