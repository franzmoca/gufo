# Gemma 4 26B-A4B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth
`gemma-4-26B-A4B-it` UD-Q4_K_XL, UD-Q6_K_XL and UD-Q8_K_XL targets with the
Unsloth Q8_0 `gemma4-assistant` MTP drafter. The UD-Q8_K_XL tables are a
reduced grid (depths to 32K, up to two users). Gufo drafts up to 15 tokens
under its calibrated length control, verifying the drafter's runner-up beside
each draft; llama.cpp drafts up to four. Gufo columns measured October 5,
2026; llama.cpp columns September 28–30. HTTP, greedy,
thinking off. llama.cpp is the repository's pinned `b11069` (ROCm) reference.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3703.22 | 1684.76 | +119.8% | 53.03 | 43.96 | +20.6% |
| 4,096 | 3325.87 | 1404.62 | +136.8% | 52.50 | 43.26 | +21.4% |
| 8,192 | 2896.22 | 1251.89 | +131.3% | 52.25 | 42.58 | +22.7% |
| 12,288 | 2658.16 | 1115.27 | +138.3% | 51.63 | 41.94 | +23.1% |
| 16,384 | 2373.17 | 1022.60 | +132.1% | 50.14 | 41.42 | +21.1% |
| 32,768 | 2069.08 | 746.17 | +177.3% | 48.07 | 38.57 | +24.6% |
| 65,536 | 1367.62 | 472.56 | +189.4% | 44.23 | 35.12 | +25.9% |
| 131,072 | 823.42 | 281.37 | +192.6% | 37.97 | 29.42 | +29.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3531.52 | 1425.52 | +147.7% | 48.74 | 40.94 | +19.1% |
| 4,096 | 3194.05 | 1215.03 | +162.9% | 48.21 | 40.33 | +19.5% |
| 8,192 | 2775.43 | 1103.41 | +151.5% | 47.97 | 39.75 | +20.7% |
| 12,288 | 2597.14 | 996.77 | +160.6% | 47.44 | 39.19 | +21.1% |
| 16,384 | 2290.72 | 918.82 | +149.3% | 46.18 | 38.75 | +19.2% |
| 32,768 | 1976.84 | 679.89 | +190.8% | 44.41 | 36.64 | +21.2% |
| 65,536 | 1320.99 | 447.13 | +195.4% | 41.11 | 33.19 | +23.9% |
| 131,072 | 812.93 | 268.45 | +202.8% | 35.65 | 27.98 | +27.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q6.svg)

---

<!-- bench:single-ar-q8 -->
| Gemma 4 26B-A4B UD-Q8_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3465.59 | 1392.01 | +149.0% | 45.79 | 38.69 | +18.4% |
| 4,096 | 3138.80 | 1212.07 | +159.0% | 45.33 | 38.17 | +18.8% |
| 16,384 | 2350.55 | 907.60 | +159.0% | 43.53 | 36.66 | +18.7% |
| 32,768 | 1986.29 | 680.43 | +191.9% | 41.95 | 34.84 | +20.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text. Greedy texts differ between the engines, so accepted drafts per cycle
differ per depth (artifacts). Over the same HTTP path Gufo's AR decodes at
50.7 (Q4), 47.0 (Q6) and 45.7 (Q8) tok/s at d0.

<!-- bench:single-mtp-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3698.15 | 1630.60 | +126.8% | 79.63 | 57.61 | +38.2% | 252.78 | 62.06 | +307.3% |
| 4,096 | 3325.64 | 1351.13 | +146.1% | 84.24 | 58.64 | +43.7% | 213.80 | 65.36 | +227.1% |
| 8,192 | 2914.28 | 1199.20 | +143.0% | 78.83 | 56.85 | +38.7% | 274.10 | 39.35 | +596.6% |
| 12,288 | 2713.42 | 1051.25 | +158.1% | 80.85 | 56.72 | +42.5% | 233.86 | 45.61 | +412.7% |
| 16,384 | 2623.52 | 957.77 | +173.9% | 65.18 | 49.12 | +32.7% | 198.85 | 57.46 | +246.1% |
| 32,768 | 2022.48 | 704.89 | +186.9% | 66.82 | 39.98 | +67.1% | 217.57 | 61.27 | +255.1% |
| 65,536 | 1349.21 | 461.26 | +192.5% | 69.61 | 31.41 | +121.6% | 162.01 | 32.73 | +395.0% |
| 131,072 | 829.89 | 276.99 | +199.6% | 48.25 | 20.98 | +130.0% | 120.40 | 22.16 | +443.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4.svg)

---

<!-- bench:single-mtp-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3541.63 | 1393.42 | +154.2% | 70.76 | 54.03 | +31.0% | 250.27 | 58.75 | +326.0% |
| 4,096 | 3200.47 | 1183.58 | +170.4% | 72.11 | 54.17 | +33.1% | 198.71 | 63.89 | +211.0% |
| 8,192 | 2797.68 | 1043.91 | +168.0% | 74.90 | 50.06 | +49.6% | 255.66 | 39.41 | +548.7% |
| 12,288 | 2602.90 | 943.21 | +176.0% | 76.82 | 54.67 | +40.5% | 216.05 | 58.98 | +266.3% |
| 16,384 | 2562.32 | 864.65 | +196.3% | 74.69 | 45.14 | +65.5% | 177.29 | 52.93 | +235.0% |
| 32,768 | 1973.88 | 651.38 | +203.0% | 60.89 | 37.61 | +61.9% | 213.85 | 60.24 | +255.0% |
| 65,536 | 1325.25 | 435.03 | +204.6% | 57.34 | 33.29 | +72.2% | 154.46 | 31.08 | +397.0% |
| 131,072 | 820.01 | 265.20 | +209.2% | 45.33 | 22.62 | +100.4% | 116.07 | 15.26 | +660.6% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q6.svg)

---

<!-- bench:single-mtp-q8 -->
| Gemma 4 26B-A4B UD-Q8_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3489.85 | 1376.85 | +153.5% | 70.41 | 47.09 | +49.5% | 240.09 | 54.45 | +340.9% |
| 4,096 | 3166.65 | 1175.04 | +169.5% | 68.83 | 49.15 | +40.0% | 194.87 | 58.39 | +233.7% |
| 16,384 | 2377.91 | 885.18 | +168.6% | 59.95 | 47.25 | +26.9% | 187.92 | 54.67 | +243.7% |
| 32,768 | 1982.67 | 659.89 | +200.5% | 63.33 | 34.99 | +81.0% | 184.52 | 58.45 | +215.7% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q8.svg)

## Single user, sampled MTP

Temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05; a story-writing turn
after the cached prefix. Mean of five requests with seeds 1–5. Seed 3's prompt
is 2051 tokens; a prefill chunk past 2048 tokens runs about 5% slower, which
widens Gufo's prefill spread.

<!-- bench:single-mtp-sampled-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3566.29 ± 194.34 | 1580.72 ± 16.34 | +125.6% | 72.71 ± 2.78 | 51.02 ± 3.74 | +42.5% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q4.svg)

---

<!-- bench:single-mtp-sampled-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3413.76 ± 185.85 | 1360.32 ± 10.89 | +151.0% | 68.07 ± 0.80 | 45.05 ± 2.29 | +51.1% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q6.svg)

---

<!-- bench:single-mtp-sampled-q8 -->
| Gemma 4 26B-A4B UD-Q8_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3359.35 ± 204.60 | 1364.58 ± 5.04 | +146.2% | 60.05 ± 0.72 | 39.46 ± 3.53 | +52.2% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q8.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 53.16 | 43.99 | +20.8% |
| 2 | 92.38 | 71.85 | +28.6% |
| 4 | 150.05 | 106.17 | +41.3% |
| 6 | 182.53 | 129.53 | +40.9% |
| 8 | 213.40 | 146.37 | +45.8% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 48.74 | 40.76 | +19.6% |
| 2 | 86.44 | 66.71 | +29.6% |
| 4 | 143.94 | 95.56 | +50.6% |
| 6 | 176.96 | 122.78 | +44.1% |
| 8 | 206.97 | 133.78 | +54.7% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q6.svg)

---

<!-- bench:multi-ar-q8 -->
| Gemma 4 26B-A4B UD-Q8_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 45.79 | 38.46 | +19.1% |
| 2 | 82.18 | 59.11 | +39.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128. C1 directly
cross-checks that row. All sessions prefilled before timed decoding. Gufo
prices each session's drafts against the whole batched forward, so at C8 it
decodes about as fast as batched AR; UD-Q6_K_XL mixed text at C4 still trails
Gufo's own AR (126 vs 143 tok/s, one sample).

<!-- bench:multi-mtp-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 79.09 | 56.96 | +38.9% | 260.70 | 61.64 | +322.9% |
| 2 | 135.48 | 89.93 | +50.7% | 308.93 | 102.28 | +202.0% |
| 4 | 182.97 | 126.24 | +44.9% | 307.70 | 140.44 | +119.1% |
| 6 | 193.30 | 160.35 | +20.5% | 335.53 | 171.28 | +95.9% |
| 8 | 221.11 | 184.92 | +19.6% | 281.11 | 196.02 | +43.4% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4.svg)

---

<!-- bench:multi-mtp-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 71.55 | 51.22 | +39.7% | 253.64 | 57.94 | +337.8% |
| 2 | 115.68 | 81.47 | +42.0% | 270.94 | 90.28 | +200.1% |
| 4 | 150.32 | 113.95 | +31.9% | 293.60 | 131.20 | +123.8% |
| 6 | 179.82 | 152.44 | +18.0% | 317.84 | 167.68 | +89.6% |
| 8 | 194.26 | 170.86 | +13.7% | 271.56 | 197.08 | +37.8% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q6.svg)

---

<!-- bench:multi-mtp-q8 -->
| Gemma 4 26B-A4B UD-Q8_K_XL MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 71.36 | 47.59 | +49.9% | 239.15 | 52.22 | +358.0% |
| 2 | 120.75 | 75.70 | +59.5% | 269.29 | 88.95 | +202.7% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q8.svg)

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
| UD-Q4_K_XL | 3.90 | 5.63 | +44.4% |
| UD-Q6_K_XL | 4.91 | 6.84 | +39.3% |
| UD-Q8_K_XL | 5.67 | 7.65 | +34.9% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Context capacity 133121, autoregressive. llama.cpp preallocates its KV
cache, so its footprint does not grow with the prefix. Gufo's figure includes
its prompt cache snapshots (see the
[31B](../gemma-4-31b/BENCHMARKS.md#memory-occupation)); llama.cpp runs with
`--cache-ram 0`.

<!-- bench:memory-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 22.81 | 23.64 | +3.6% |
| 16K prefix, pp4096 + tg128 | 26.00 | 24.18 | -7.0% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4.svg)

---

<!-- bench:memory-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 28.65 | 29.49 | +2.9% |
| 16K prefix, pp4096 + tg128 | 31.84 | 30.03 | -5.7% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q6.svg)

---

<!-- bench:memory-q8 -->
| Gemma 4 26B-A4B UD-Q8_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 32.72 | 33.59 | +2.7% |
| 16K prefix, pp4096 + tg128 | 35.91 | 34.14 | -4.9% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q8.svg)
