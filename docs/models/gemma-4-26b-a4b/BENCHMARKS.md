# Gemma 4 26B-A4B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth
`gemma-4-26B-A4B-it` UD-Q4_K_XL and UD-Q6_K_XL targets with the Unsloth Q8_0
`gemma4-assistant` MTP drafter. Gufo drafts up to seven tokens under its
calibrated length control; llama.cpp drafts up to four. HTTP, greedy,
thinking off. llama.cpp is the repository's pinned `b11069` (ROCm) reference.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3588.89 | 1684.76 | +113.0% | 51.13 | 43.96 | +16.3% |
| 4,096 | 3282.56 | 1404.62 | +133.7% | 50.61 | 43.26 | +17.0% |
| 8,192 | 2961.47 | 1251.89 | +136.6% | 50.34 | 42.58 | +18.2% |
| 12,288 | 2689.62 | 1115.27 | +141.2% | 49.72 | 41.94 | +18.6% |
| 16,384 | 2572.62 | 1022.60 | +151.6% | 48.31 | 41.42 | +16.6% |
| 32,768 | 2035.30 | 746.17 | +172.8% | 46.35 | 38.57 | +20.2% |
| 65,536 | 1389.12 | 472.56 | +194.0% | 42.78 | 35.12 | +21.8% |
| 131,072 | 863.82 | 281.37 | +207.0% | 36.88 | 29.42 | +25.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3418.47 | 1425.52 | +139.8% | 47.33 | 40.94 | +15.6% |
| 4,096 | 3132.38 | 1215.03 | +157.8% | 46.88 | 40.33 | +16.2% |
| 8,192 | 2807.64 | 1103.41 | +154.5% | 46.65 | 39.75 | +17.4% |
| 12,288 | 2598.76 | 996.77 | +160.7% | 46.10 | 39.19 | +17.6% |
| 16,384 | 2521.24 | 918.82 | +174.4% | 44.94 | 38.75 | +16.0% |
| 32,768 | 1971.85 | 679.89 | +190.0% | 43.19 | 36.64 | +17.9% |
| 65,536 | 1337.39 | 447.13 | +199.1% | 40.05 | 33.19 | +20.7% |
| 131,072 | 849.64 | 268.45 | +216.5% | 34.82 | 27.98 | +24.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q6.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text. Greedy texts differ between the engines, so accepted drafts per cycle
differ per depth (artifacts). Over the same HTTP path Gufo's AR decodes at
50.7 (Q4) and 47.0 (Q6) tok/s at d0.

<!-- bench:single-mtp-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3562.58 | 1630.60 | +118.5% | 78.80 | 57.61 | +36.8% | 173.74 | 62.06 | +180.0% |
| 4,096 | 3294.60 | 1351.13 | +143.8% | 80.98 | 58.64 | +38.1% | 152.57 | 65.36 | +133.4% |
| 8,192 | 2947.79 | 1199.20 | +145.8% | 76.19 | 56.85 | +34.0% | 131.37 | 39.35 | +233.9% |
| 12,288 | 2670.49 | 1051.25 | +154.0% | 80.25 | 56.72 | +41.5% | 174.16 | 45.61 | +281.8% |
| 16,384 | 2614.55 | 957.77 | +173.0% | 65.30 | 49.12 | +32.9% | 140.02 | 57.46 | +143.7% |
| 32,768 | 2074.21 | 704.89 | +194.3% | 65.11 | 39.98 | +62.9% | 142.80 | 61.27 | +133.1% |
| 65,536 | 1364.69 | 461.26 | +195.9% | 66.01 | 31.41 | +110.2% | 129.61 | 32.73 | +296.0% |
| 131,072 | 861.20 | 276.99 | +210.9% | 47.81 | 20.98 | +127.9% | 95.36 | 22.16 | +330.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4.svg)

---

<!-- bench:single-mtp-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3420.59 | 1393.42 | +145.5% | 68.91 | 54.03 | +27.5% | 187.72 | 58.75 | +219.5% |
| 4,096 | 3165.09 | 1183.58 | +167.4% | 71.50 | 54.17 | +32.0% | 139.66 | 63.89 | +118.6% |
| 8,192 | 2836.98 | 1043.91 | +171.8% | 76.78 | 50.06 | +53.4% | 140.67 | 39.41 | +256.9% |
| 12,288 | 2590.91 | 943.21 | +174.7% | 71.24 | 54.67 | +30.3% | 160.11 | 58.98 | +171.5% |
| 16,384 | 2525.22 | 864.65 | +192.1% | 71.20 | 45.14 | +57.7% | 128.56 | 52.93 | +142.9% |
| 32,768 | 2026.86 | 651.38 | +211.2% | 60.44 | 37.61 | +60.7% | 90.43 | 60.24 | +50.1% |
| 65,536 | 1339.71 | 435.03 | +208.0% | 56.61 | 33.29 | +70.1% | 116.56 | 31.08 | +275.0% |
| 131,072 | 852.58 | 265.20 | +221.5% | 44.72 | 22.62 | +97.7% | 91.17 | 15.26 | +497.4% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q6.svg)

## Single user, sampled MTP

Temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05; a story-writing turn
after the cached prefix. Mean of three requests with seeds 1–3.

<!-- bench:single-mtp-sampled-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3544.21 ± 5.34 | 1604.71 ± 6.23 | +120.9% | 63.43 ± 1.99 | 46.41 ± 1.54 | +36.7% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q4.svg)

---

<!-- bench:single-mtp-sampled-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3422.19 ± 5.80 | 1383.68 ± 1.03 | +147.3% | 58.20 ± 1.72 | 45.17 ± 3.99 | +28.8% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q6.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 50.98 | 43.99 | +15.9% |
| 2 | 92.67 | 71.85 | +29.0% |
| 4 | 150.78 | 106.17 | +42.0% |
| 6 | 183.39 | 129.53 | +41.6% |
| 8 | 213.91 | 146.37 | +46.1% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 47.35 | 40.76 | +16.2% |
| 2 | 86.77 | 66.71 | +30.1% |
| 4 | 144.47 | 95.56 | +51.2% |
| 6 | 177.48 | 122.78 | +44.6% |
| 8 | 207.01 | 133.78 | +54.7% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q6.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128. C1 directly
cross-checks that row. All sessions prefilled before timed decoding. Gufo
prices each session's drafts against the whole batched forward, so at C8 it
decodes about as fast as batched AR; UD-Q6_K_XL mixed text at C4 still trails
Gufo's own AR (126 vs 143 tok/s, one sample).

<!-- bench:multi-mtp-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 79.24 | 56.96 | +39.1% | 175.54 | 61.64 | +184.8% |
| 2 | 134.98 | 89.93 | +50.1% | 259.85 | 102.28 | +154.1% |
| 4 | 181.42 | 126.24 | +43.7% | 249.43 | 140.44 | +77.6% |
| 6 | 210.14 | 160.35 | +31.1% | 225.67 | 171.28 | +31.8% |
| 8 | 221.74 | 184.92 | +19.9% | 256.74 | 196.02 | +31.0% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4.svg)

---

<!-- bench:multi-mtp-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 69.28 | 51.22 | +35.3% | 189.93 | 57.94 | +227.8% |
| 2 | 119.53 | 81.47 | +46.7% | 246.15 | 90.28 | +172.7% |
| 4 | 152.51 | 113.95 | +33.8% | 257.18 | 131.20 | +96.0% |
| 6 | 194.80 | 152.44 | +27.8% | 223.41 | 167.68 | +33.2% |
| 8 | 197.92 | 170.86 | +15.8% | 251.65 | 197.08 | +27.7% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q6.svg)

## Image requests

UD-Q4_K_XL, single user, AR, context 16384, the method of the
[31B image requests](../gemma-4-31b/BENCHMARKS.md#image-requests): the same
four images pre-resized to multiples of 48, Gufo at `--image-tokens 280` or
`1120`, llama.cpp b11069 with `-b 2048 -ub 2048` and its default token range.
Cold: a fresh nonce precedes the image; follow-up: the next user turn,
reusing the image prefix. Median of three warmed requests, 64 output tokens,
greedy ([Gufo 280](artifacts/image-q4-gufo-280.json),
[Gufo 1120](artifacts/image-q4-gufo-1120.json),
[llama.cpp](artifacts/image-q4-reference.json); `tools/gemma4/image_bench.py`).

| Image | Budget | Prompt tokens | Gufo cold TTFT (s) | llama.cpp cold TTFT (s) | Gain | Gufo follow-up TTFT (s) | llama.cpp follow-up TTFT (s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chart 624×960 | 280 | 312 | 0.36 | 0.74 | +105.7% | 0.10 | 0.22 | +110.2% | 53.11 | 44.66 | +18.9% |
| Logo 768×768 | 280 | 309 | 0.36 | 0.74 | +108.2% | 0.10 | 0.22 | +119.0% | 53.12 | 43.12 | +23.2% |
| Chart 1296×1968 | 1120 | 1159 | 1.63 | 4.49 | +175.9% | 0.15 | 0.37 | +148.8% | 51.50 | 38.09 | +35.2% |
| Logo 1584×1584 | 1120 | 1142 | 1.63 | 4.39 | +169.9% | 0.16 | 0.39 | +138.7% | 51.53 | 36.98 | +39.4% |

Gain is llama.cpp time over Gufo time minus one (decode: Gufo over
llama.cpp). The vision encoder is the 31B's (162 ms for 260 soft tokens,
1,147 ms for 1,107); most of a cold 1,120-token request is the encoder.

## Loading time

C1, context capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 26B-A4B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 3.87 | 5.63 | +45.5% |
| UD-Q6_K_XL | 4.91 | 6.84 | +39.3% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Context capacity 133121, autoregressive. llama.cpp preallocates its KV
cache, so its footprint does not grow with the prefix.

<!-- bench:memory-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 22.80 | 23.64 | +3.7% |
| 16K prefix, pp4096 + tg128 | 23.54 | 24.18 | +2.7% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4.svg)

---

<!-- bench:memory-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 28.64 | 29.49 | +3.0% |
| 16K prefix, pp4096 + tg128 | 29.38 | 30.03 | +2.2% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q6.svg)
