# Gemma 4 31B QAT benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth
`gemma-4-31B-it-qat-UD-Q4_K_XL` target (every projection Q4_0) and its Q4_0
QAT `gemma4-assistant` MTP drafter. Gufo drafts up to seven tokens under its
confidence-based length control (`--draft-policy confidence`, the default when these tables were measured; the calibrated default is compared in [EXPERIMENTS.md](../gemma-4-31b/EXPERIMENTS.md)); llama.cpp drafts up to four. HTTP, greedy,
thinking off. llama.cpp is the repository's pinned `b11069` (ROCm) reference.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B QAT AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 451.88 | 329.96 | +36.9% | 12.07 | 10.52 | +14.7% |
| 4,096 | 428.69 | 290.26 | +47.7% | 11.91 | 10.31 | +15.5% |
| 8,192 | 408.09 | 263.49 | +54.9% | 11.72 | 10.11 | +15.9% |
| 12,288 | 390.58 | 231.95 | +68.4% | 11.63 | 9.91 | +17.4% |
| 16,384 | 381.20 | 213.47 | +78.6% | 11.46 | 9.73 | +17.8% |
| 32,768 | 326.61 | 160.99 | +102.9% | 10.94 | 9.06 | +20.8% |
| 65,536 | 250.39 | 107.53 | +132.9% | 10.02 | 7.95 | +26.0% |
| 131,072 | 163.05 | 65.93 | +147.3% | 8.55 | 6.39 | +33.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.
Greedy texts differ between the engines, so accepted drafts per cycle differ
per depth (artifacts); llama.cpp's repetitive acceptance collapses past 32K.

<!-- bench:single-mtp -->
| Gemma 4 31B QAT MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 458.57 | 322.01 | +42.4% | 25.56 | 21.22 | +20.5% | 70.89 | 37.97 | +86.7% |
| 4,096 | 434.33 | 274.16 | +58.4% | 27.26 | 22.52 | +21.0% | 66.85 | 38.44 | +73.9% |
| 8,192 | 409.86 | 247.28 | +65.7% | 23.96 | 21.39 | +12.0% | 64.50 | 34.04 | +89.5% |
| 12,288 | 390.53 | 225.34 | +73.3% | 21.53 | 18.46 | +16.6% | 63.23 | 32.02 | +97.5% |
| 16,384 | 380.95 | 207.50 | +83.6% | 24.84 | 20.13 | +23.4% | 59.25 | 31.57 | +87.7% |
| 32,768 | 332.49 | 156.96 | +111.8% | 22.14 | 14.76 | +50.0% | 53.69 | 17.62 | +204.7% |
| 65,536 | 253.81 | 106.01 | +139.4% | 19.09 | 10.94 | +74.5% | 38.00 | 9.60 | +295.8% |
| 131,072 | 163.22 | 64.90 | +151.5% | 12.66 | 6.69 | +89.2% | 20.97 | 3.85 | +444.7% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Single user, sampled MTP

Temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05; a story-writing turn
after the cached prefix. Mean of three requests with seeds 1–3. At 0–4K
Gufo's cycle is cheaper (105–109 vs 114–122 ms) but its confidence floor
proposes 2.4 drafts per cycle against llama.cpp's fixed four, so it accepts
fewer per cycle (1.0–1.3 vs 1.5); the floor was tuned on the standard 31B's
Q8_0 drafter. The 4K gap is within one standard deviation.

<!-- bench:single-mtp-sampled -->
| Gemma 4 31B QAT MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 458.80 ± 5.36 | 313.73 ± 2.00 | +46.2% | 22.05 ± 1.05 | 21.79 ± 0.63 | +1.2% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar -->
| Gemma 4 31B QAT AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.08 | 10.50 | +15.0% |
| 2 | 21.74 | 18.99 | +14.5% |
| 4 | 37.80 | 32.10 | +17.8% |
| 6 | 48.75 | 40.95 | +19.0% |
| 8 | 54.79 | 45.30 | +20.9% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. As on the standard 31B, Gufo verifies all sessions'
drafts in one forward of at most 16 exact rows (16 / users − 1 drafts per
session), while llama.cpp verifies up to four drafts per user with
Q8_1-activation matrix kernels that scale further with rows.

<!-- bench:multi-mtp -->
| Gemma 4 31B QAT MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.55 | 24.36 | +4.9% | 71.30 | 39.61 | +80.0% |
| 2 | 41.19 | 38.18 | +7.9% | 81.26 | 60.58 | +34.1% |
| 4 | 53.99 | 37.44 | +44.2% | 73.04 | 66.07 | +10.5% |
| 6 | 59.64 | 49.68 | +20.0% | 65.09 | 87.51 | -25.6% |
| 8 | 61.70 | 61.50 | +0.3% | 68.21 | 100.93 | -32.4% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## MTP draft policy

`--draft-policy calibrated` became the default on September 28, 2026; the MTP
tables above use it. Relative A/B of the two policies on one development
build (`gpu-test` preset), two runs each in ABBA order with one server per
run; greedy texts are identical under both policies, sampled texts differ
because draft counts change the random draws. Depth is a shared earlier
conversation turn reused from the prompt cache.
[d0](artifacts/draft-policy-ab-d0k.json) · [8K](artifacts/draft-policy-ab-d8k.json) ·
`tools/gemma4/draft_policy_bench.py`

| Gemma 4 31B QAT MTP<br>Workload | d0 confidence (tok/s) | d0 calibrated (tok/s) | Gain | 8K confidence (tok/s) | 8K calibrated (tok/s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose, greedy (12 prompts) | 25.26 | 25.98 | +2.9% | 23.45 | 24.04 | +2.5% |
| Story, temperature 1 (12 seeds) | 22.11 | 22.48 | +1.7% | 20.82 | 21.17 | +1.7% |
| Code writing, greedy | 42.17 | 42.99 | +2.0% | 38.04 | 37.95 | -0.2% |
| Code writing, temperature 0.6 (12 seeds) | 30.82 | 32.36 | +5.0% | 27.89 | 29.04 | +4.1% |
| Code edit in context, greedy | 62.79 | 62.58 | -0.3% | 59.84 | 59.36 | -0.8% |
| Repetitive, greedy | 66.82 | 64.60 | -3.3% | 62.34 | 61.49 | -1.4% |

Prose and new code gain; verbatim copying is within about 1.5% (the QAT
d0 repetitive cell includes one slow calibrated run). Per-run values and
draft counts are in the artifacts.

## Loading time

C1, capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 31B QAT<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| QAT | 3.98 | 6.46 | +62.3% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 133121 tokens, AR. Both engines allocate KV for the whole capacity;
device total minus free on this unified-memory APU.

<!-- bench:memory -->
| Gemma 4 31B QAT AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 31.52 | 33.54 | +6.4% |
| 16K prefix, pp4096 + tg128 | 34.44 | 35.71 | +3.7% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
