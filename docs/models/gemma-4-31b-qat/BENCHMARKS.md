# Gemma 4 31B QAT benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth
`gemma-4-31B-it-qat-UD-Q4_K_XL` target (every projection Q4_0) and its Q4_0
QAT `gemma4-assistant` MTP drafter. Gufo drafts up to 15 tokens under its
calibrated length control, verifying the drafter's runner-up beside each
draft; llama.cpp drafts up to four. Gufo columns measured October 5, 2026;
llama.cpp columns September 28. HTTP, greedy, thinking off. llama.cpp is the repository's pinned `b11069` (ROCm) reference.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B QAT AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 561.18 | 329.96 | +70.1% | 12.01 | 10.52 | +14.2% |
| 4,096 | 524.10 | 290.26 | +80.6% | 11.86 | 10.31 | +15.0% |
| 8,192 | 483.16 | 263.49 | +83.4% | 11.65 | 10.11 | +15.2% |
| 12,288 | 447.41 | 231.95 | +92.9% | 11.56 | 9.91 | +16.6% |
| 16,384 | 392.27 | 213.47 | +83.8% | 11.37 | 9.73 | +16.9% |
| 32,768 | 361.31 | 160.99 | +124.4% | 10.84 | 9.06 | +19.6% |
| 65,536 | 268.98 | 107.53 | +150.1% | 9.92 | 7.95 | +24.8% |
| 131,072 | 180.90 | 65.93 | +174.4% | 8.44 | 6.39 | +32.1% |
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
| 0 | 562.87 | 322.01 | +74.8% | 25.77 | 21.22 | +21.4% | 122.63 | 37.97 | +223.0% |
| 4,096 | 518.47 | 274.16 | +89.1% | 27.26 | 22.52 | +21.0% | 114.91 | 38.44 | +198.9% |
| 8,192 | 468.59 | 247.28 | +89.5% | 23.60 | 21.39 | +10.3% | 107.26 | 34.04 | +215.1% |
| 12,288 | 437.98 | 225.34 | +94.4% | 21.14 | 18.46 | +14.5% | 104.14 | 32.02 | +225.2% |
| 16,384 | 429.46 | 207.50 | +107.0% | 23.32 | 20.13 | +15.8% | 92.28 | 31.57 | +192.3% |
| 32,768 | 359.54 | 156.96 | +129.1% | 21.42 | 14.76 | +45.1% | 78.30 | 17.62 | +344.4% |
| 65,536 | 263.84 | 106.01 | +148.9% | 17.53 | 10.94 | +60.2% | 48.61 | 9.60 | +406.4% |
| 131,072 | 180.03 | 64.90 | +177.4% | 12.56 | 6.69 | +87.7% | 21.36 | 3.85 | +454.8% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Single user, sampled MTP

Temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05; a story-writing turn
after the cached prefix. Mean of five requests with seeds 1–5. Both engines
accept about as many drafts per request (70–78 vs 69–76), but Gufo's calibrated
draft length proposes fewer (160–190 vs 198–225; 43% vs 34% accepted), so less
of each cycle verifies drafts that are rejected.

<!-- bench:single-mtp-sampled -->
| Gemma 4 31B QAT MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 546.56 ± 26.87 | 317.38 ± 2.46 | +72.2% | 24.06 ± 1.97 | 19.84 ± 1.03 | +21.3% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar -->
| Gemma 4 31B QAT AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.01 | 10.50 | +14.4% |
| 2 | 21.60 | 18.99 | +13.7% |
| 4 | 37.57 | 32.10 | +17.0% |
| 6 | 47.77 | 40.95 | +16.7% |
| 8 | 57.81 | 45.30 | +27.6% |
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
| 1 | 25.62 | 24.36 | +5.2% | 120.03 | 39.61 | +203.0% |
| 2 | 42.48 | 38.18 | +11.3% | 128.67 | 60.58 | +112.4% |
| 4 | 67.52 | 37.44 | +80.3% | 123.58 | 66.07 | +87.0% |
| 6 | 79.24 | 49.68 | +59.5% | 136.52 | 87.51 | +56.0% |
| 8 | 82.55 | 61.50 | +34.2% | 105.44 | 100.93 | +4.5% |
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
| QAT | 3.95 | 6.46 | +63.5% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 133121 tokens, AR. Both engines allocate KV for the whole capacity;
device total minus free on this unified-memory APU. Gufo's figure includes its
prompt cache snapshots (see the
[standard 31B](../gemma-4-31b/BENCHMARKS.md#memory-occupation)); llama.cpp runs
with `--cache-ram 0`.

<!-- bench:memory -->
| Gemma 4 31B QAT AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 31.60 | 33.54 | +6.1% |
| 16K prefix, pp4096 + tg128 | 44.38 | 35.71 | -19.5% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
