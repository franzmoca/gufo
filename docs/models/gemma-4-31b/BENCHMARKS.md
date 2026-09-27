# Gemma 4 31B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target
and its `gemma4-assistant` Q8_0 MTP drafter, four draft tokens on both engines.
HTTP, greedy, thinking off. llama.cpp is the repository's pinned `b11069`
(ROCm) reference for AR and MTP.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 404.47 | 300.91 | +34.4% | 10.82 | 9.80 | +10.4% |
| 4,096 | 369.61 | 264.62 | +39.7% | 10.70 | 9.61 | +11.3% |
| 8,192 | 345.96 | 239.48 | +44.5% | 10.55 | 9.43 | +11.9% |
| 12,288 | 320.84 | 215.96 | +48.6% | 10.45 | 9.26 | +12.9% |
| 16,384 | 315.07 | 202.00 | +56.0% | 10.32 | 9.11 | +13.3% |
| 32,768 | 264.18 | 154.12 | +71.4% | 9.89 | 8.51 | +16.2% |
| 65,536 | 197.81 | 104.28 | +89.7% | 9.13 | 7.53 | +21.2% |
| 131,072 | 134.35 | 64.56 | +108.1% | 7.90 | 6.12 | +29.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-mtp -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 398.31 | 294.20 | +35.4% | 22.65 | 19.08 | +18.7% | 36.34 | 31.11 | +16.8% |
| 4,096 | 371.29 | 253.33 | +46.6% | 22.34 | 20.32 | +9.9% | 38.84 | 32.28 | +20.3% |
| 8,192 | 341.84 | 231.63 | +47.6% | 21.18 | 18.58 | +14.0% | 35.69 | 28.74 | +24.2% |
| 12,288 | 320.72 | 211.21 | +51.8% | 21.48 | 18.14 | +18.4% | 36.18 | 28.37 | +27.5% |
| 16,384 | 314.76 | 195.22 | +61.2% | 20.16 | 15.95 | +26.4% | 35.30 | 27.19 | +29.8% |
| 32,768 | 263.83 | 150.59 | +75.2% | 20.27 | 14.89 | +36.1% | 33.72 | 21.08 | +60.0% |
| 65,536 | 197.48 | 102.73 | +92.2% | 15.54 | 10.56 | +47.2% | 28.13 | 14.85 | +89.4% |
| 131,072 | 134.05 | 63.79 | +110.1% | 12.44 | 6.92 | +79.8% | 21.60 | 4.09 | +428.1% |
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
| 0 | 393.19 ± 1.86 | 293.21 ± 0.96 | +34.1% | 19.84 ± 0.92 | 18.12 ± 1.97 | +9.5% |
| 4,096 | 365.31 ± 2.51 | 256.40 ± 2.11 | +42.5% | 18.49 ± 2.45 | 16.51 ± 0.64 | +12.0% |
| 16,384 | 308.22 ± 2.63 | 195.75 ± 1.13 | +57.5% | 17.95 ± 1.18 | 14.03 ± 1.35 | +27.9% |
| 32,768 | 261.85 ± 3.56 | 150.21 ± 0.25 | +74.3% | 16.04 ± 1.41 | 10.62 ± 0.56 | +51.0% |
| 65,536 | 197.77 ± 0.61 | 102.75 ± 0.08 | +92.5% | 15.10 ± 0.91 | 9.37 ± 0.88 | +61.2% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
Gufo serves Gemma 4 requests one at a time (no batched decode yet), so its
summed per-request rates would overstate throughput; its cells stay TODO.

<!-- bench:multi-ar -->
| Gemma 4 31B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | TODO | 9.79 | TODO |
| 2 | TODO | 17.54 | TODO |
| 4 | TODO | 28.83 | TODO |
| 6 | TODO | 34.03 | TODO |
| 8 | TODO | 35.08 | TODO |
<!-- /bench -->

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Gemma 4 31B Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TODO | 20.73 | TODO | TODO | 30.93 | TODO |
| 2 | TODO | 32.25 | TODO | TODO | 49.44 | TODO |
| 4 | TODO | 48.28 | TODO | TODO | 75.05 | TODO |
| 6 | TODO | 58.11 | TODO | TODO | 99.65 | TODO |
| 8 | TODO | 57.05 | TODO | TODO | 87.82 | TODO |
<!-- /bench -->


## gemma-control fork (single user)

The previous production setup: halo-box/strix-llama.cpp `8c1c282ec` on Vulkan
RADV with `GGML_VK_MMV_NO_SPLIT=1 -b 2048 -ub 512`, same GGUF files, same
driver workloads and four draft tokens. Hand-rendered from
[artifacts/fork](artifacts/fork). At 128K with MTP its Vulkan queue timed out
(`Fence fallback timer expired on ring comp_1.1.0`); its repetitive MTP
workload was not run.

Greedy MTP texts differ between the engines, so accepted drafts per cycle
differ per depth; at 4K the fork accepts 1.91 drafts per cycle against
Gufo's 1.67. Gufo's verification cycle (draft plus verify) is shorter at every
measured depth: 115 vs 120 ms at d0, 120 vs 126 ms at 4K, 137 vs 165 ms at
32K and 158 vs 211 ms at 64K.

| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 404.47 | 293.92 | +37.6% | 10.82 | 10.53 | +2.8% |
| 4,096 | 369.61 | 259.83 | +42.3% | 10.70 | 10.29 | +4.0% |
| 8,192 | 345.96 | 243.79 | +41.9% | 10.55 | 10.01 | +5.4% |
| 12,288 | 320.84 | 220.65 | +45.4% | 10.45 | 9.87 | +5.9% |
| 16,384 | 315.07 | 211.76 | +48.8% | 10.32 | 9.58 | +7.7% |
| 32,768 | 264.18 | 164.54 | +60.6% | 9.89 | 8.93 | +10.8% |
| 65,536 | 197.81 | 117.97 | +67.7% | 9.13 | 7.74 | +18.0% |
| 131,072 | 134.35 | 74.00 | +81.6% | 7.90 | 6.10 | +29.5% |

---

| Gemma 4 31B Q4 MTP mixed<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain | Gufo / fork accepted per cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 390.45 | 286.71 | +36.2% | 22.65 | 21.71 | +4.3% | 1.61 / 1.61 |
| 4,096 | 370.23 | 253.75 | +45.9% | 22.34 | 23.17 | -3.6% | 1.67 / 1.91 |
| 8,192 | 340.47 | 234.69 | +45.1% | 21.18 | 20.84 | +1.6% | 1.56 / 1.78 |
| 12,288 | 320.24 | 216.64 | +47.8% | 21.48 | 17.49 | +22.8% | 1.67 / 1.37 |
| 16,384 | 314.76 | 201.12 | +56.5% | 20.16 | 19.42 | +3.8% | 1.56 / 1.72 |
| 32,768 | 263.83 | 159.66 | +65.2% | 20.27 | 16.47 | +23.1% | 1.78 / 1.72 |
| 65,536 | 197.48 | 114.11 | +73.1% | 15.54 | 12.67 | +22.7% | 1.46 / 1.67 |
| 131,072 | 133.94 | N/A | N/A | 12.44 | N/A | N/A | 1.51 / N/A |

## Loading time

C1, capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 31B Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 4.16 | 6.58 | +58.2% |
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
