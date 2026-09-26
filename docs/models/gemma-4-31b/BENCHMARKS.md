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
| 0 | 392.19 | 300.91 | +30.3% | 10.57 | 9.80 | +7.9% |
| 4,096 | 364.46 | 264.62 | +37.7% | 10.41 | 9.61 | +8.3% |
| 8,192 | 336.45 | 239.48 | +40.5% | 10.25 | 9.43 | +8.7% |
| 12,288 | 311.64 | 215.96 | +44.3% | 9.93 | 9.26 | +7.2% |
| 16,384 | 307.71 | 202.00 | +52.3% | 9.84 | 9.11 | +8.0% |
| 32,768 | 257.25 | 154.12 | +66.9% | 9.15 | 8.51 | +7.5% |
| 65,536 | 190.86 | 104.28 | +83.0% | 8.12 | 7.53 | +7.8% |
| 131,072 | 128.32 | 64.56 | +98.8% | 6.57 | 6.12 | +7.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-mtp -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 400.37 | 294.20 | +36.1% | 20.72 | 19.08 | +8.6% | 33.34 | 31.11 | +7.2% |
| 4,096 | 367.21 | 253.33 | +45.0% | 19.79 | 20.32 | -2.6% | 34.37 | 32.28 | +6.5% |
| 8,192 | 336.37 | 231.63 | +45.2% | 18.77 | 18.58 | +1.0% | 30.61 | 28.74 | +6.5% |
| 12,288 | 313.11 | 211.21 | +48.2% | 17.88 | 18.14 | -1.4% | 30.28 | 28.37 | +6.7% |
| 16,384 | 308.37 | 195.22 | +58.0% | 16.54 | 15.95 | +3.7% | 29.03 | 27.19 | +6.8% |
| 32,768 | 257.14 | 150.59 | +70.8% | 15.32 | 14.89 | +2.9% | 25.52 | 21.08 | +21.1% |
| 65,536 | 190.84 | 102.73 | +85.8% | 10.70 | 10.56 | +1.3% | 19.19 | 14.85 | +29.2% |
| 131,072 | 128.28 | 63.79 | +101.1% | 7.26 | 6.92 | +4.9% | 12.73 | 4.09 | +211.2% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

### Focused MTP retest after parallel draft argmax

September 27 release build, same single-user prose workload and four drafts.
The run completed through 64K and was stopped before its 128K row; repetitive
text was not remeasured. The full table above remains the prior complete sweep.
Accepted drafts per cycle were unchanged at each completed depth. These are
single samples, so small differences may include measurement noise.

| Depth | Gufo mixed (tok/s) | Gain vs prior Gufo | Gain vs llama.cpp | Gain vs fork |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 21.38 | +3.2% | +12.1% | -1.5% |
| 4,096 | 20.09 | +1.5% | -1.1% | -13.3% |
| 8,192 | 19.16 | +2.1% | +3.1% | -8.1% |
| 12,288 | 18.03 | +0.8% | -0.6% | +3.1% |
| 16,384 | 16.66 | +0.7% | +4.5% | -14.2% |
| 32,768 | 15.42 | +0.7% | +3.6% | -6.4% |
| 65,536 | 10.75 | +0.5% | +1.8% | -15.2% |

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

| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 392.19 | 293.92 | +33.4% | 10.57 | 10.53 | +0.4% |
| 4,096 | 364.46 | 259.83 | +40.3% | 10.41 | 10.29 | +1.2% |
| 8,192 | 336.45 | 243.79 | +38.0% | 10.25 | 10.01 | +2.4% |
| 12,288 | 311.64 | 220.65 | +41.2% | 9.93 | 9.87 | +0.6% |
| 16,384 | 307.71 | 211.76 | +45.3% | 9.84 | 9.58 | +2.7% |
| 32,768 | 257.25 | 164.54 | +56.3% | 9.15 | 8.93 | +2.5% |
| 65,536 | 190.86 | 117.97 | +61.8% | 8.12 | 7.74 | +4.9% |
| 131,072 | 128.32 | 74.00 | +73.4% | 6.57 | 6.10 | +7.7% |

---

| Gemma 4 31B Q4 MTP mixed<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 365.19 | 286.71 | +27.4% | 20.72 | 21.71 | -4.6% |
| 4,096 | 362.76 | 253.75 | +43.0% | 19.79 | 23.17 | -14.6% |
| 8,192 | 333.97 | 234.69 | +42.3% | 18.77 | 20.84 | -9.9% |
| 12,288 | 312.19 | 216.64 | +44.1% | 17.88 | 17.49 | +2.2% |
| 16,384 | 308.37 | 201.12 | +53.3% | 16.54 | 19.42 | -14.8% |
| 32,768 | 256.78 | 159.66 | +60.8% | 15.32 | 16.47 | -7.0% |
| 65,536 | 190.84 | 114.11 | +67.2% | 10.70 | 12.67 | -15.5% |
| 131,072 | 127.37 | N/A | N/A | 7.26 | N/A | N/A |

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
