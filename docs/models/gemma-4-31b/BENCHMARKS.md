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
| 0 | 397.79 | 300.91 | +32.2% | 10.81 | 9.80 | +10.3% |
| 4,096 | 367.31 | 264.62 | +38.8% | 10.61 | 9.61 | +10.4% |
| 8,192 | 337.70 | 239.48 | +41.0% | 10.45 | 9.43 | +10.8% |
| 12,288 | 316.13 | 215.96 | +46.4% | 10.28 | 9.26 | +11.0% |
| 16,384 | 310.86 | 202.00 | +53.9% | 10.12 | 9.11 | +11.1% |
| 32,768 | 259.22 | 154.12 | +68.2% | 9.47 | 8.51 | +11.3% |
| 65,536 | 192.46 | 104.28 | +84.6% | 8.43 | 7.53 | +12.0% |
| 131,072 | 129.86 | 64.56 | +101.1% | 6.88 | 6.12 | +12.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-mtp -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 407.09 | 294.20 | +38.4% | 23.09 | 19.08 | +21.0% | 36.25 | 31.11 | +16.5% |
| 4,096 | 375.32 | 253.33 | +48.2% | 22.40 | 20.32 | +10.2% | 38.81 | 32.28 | +20.2% |
| 8,192 | 348.32 | 231.63 | +50.4% | 22.15 | 18.58 | +19.2% | 35.70 | 28.74 | +24.2% |
| 12,288 | 317.32 | 211.21 | +50.2% | 21.36 | 18.14 | +17.8% | 36.10 | 28.37 | +27.2% |
| 16,384 | 310.94 | 195.22 | +59.3% | 20.05 | 15.95 | +25.7% | 35.19 | 27.19 | +29.4% |
| 32,768 | 259.98 | 150.59 | +72.6% | 20.15 | 14.89 | +35.3% | 33.54 | 21.08 | +59.1% |
| 65,536 | 192.55 | 102.73 | +87.4% | 15.54 | 10.56 | +47.2% | 27.78 | 14.85 | +87.1% |
| 131,072 | 129.95 | 63.79 | +103.7% | 12.08 | 6.92 | +74.6% | 20.97 | 4.09 | +412.7% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

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
Gufo's 1.67, yet Gufo's cycle is shorter (119 vs 126 ms).

| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 397.79 | 293.92 | +35.3% | 10.81 | 10.53 | +2.7% |
| 4,096 | 367.31 | 259.83 | +41.4% | 10.61 | 10.29 | +3.1% |
| 8,192 | 337.70 | 243.79 | +38.5% | 10.45 | 10.01 | +4.4% |
| 12,288 | 316.13 | 220.65 | +43.3% | 10.28 | 9.87 | +4.2% |
| 16,384 | 310.86 | 211.76 | +46.8% | 10.12 | 9.58 | +5.6% |
| 32,768 | 259.22 | 164.54 | +57.5% | 9.47 | 8.93 | +6.0% |
| 65,536 | 192.46 | 117.97 | +63.1% | 8.43 | 7.74 | +8.9% |
| 131,072 | 129.86 | 74.00 | +75.5% | 6.88 | 6.10 | +12.8% |

---

| Gemma 4 31B Q4 MTP mixed<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain | Gufo / fork accepted per cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 407.09 | 286.71 | +42.0% | 23.09 | 21.71 | +6.4% | 1.67 / 1.61 |
| 4,096 | 375.32 | 253.75 | +47.9% | 22.40 | 23.17 | -3.3% | 1.67 / 1.91 |
| 8,192 | 348.32 | 234.69 | +48.4% | 22.15 | 20.84 | +6.3% | 1.67 / 1.78 |
| 12,288 | 316.38 | 216.64 | +46.0% | 21.36 | 17.49 | +22.1% | 1.67 / 1.37 |
| 16,384 | 310.94 | 201.12 | +54.6% | 20.05 | 19.42 | +3.2% | 1.56 / 1.72 |
| 32,768 | 259.98 | 159.66 | +62.8% | 20.15 | 16.47 | +22.3% | 1.78 / 1.72 |
| 65,536 | 192.55 | 114.11 | +68.7% | 15.54 | 12.67 | +22.7% | 1.51 / 1.67 |
| 131,072 | 129.95 | N/A | N/A | 12.08 | N/A | N/A | 1.51 / N/A |

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
minus free, which on this unified-memory APU tracks system-wide use; compare
the two engines rather than the absolute values.

<!-- bench:memory -->
| Gemma 4 31B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 60.30 | 58.94 | -2.3% |
| 16K prefix, pp4096 + tg128 | 62.59 | 61.11 | -2.4% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
