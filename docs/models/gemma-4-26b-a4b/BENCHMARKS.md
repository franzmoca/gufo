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
| 0 | 3137.73 | 1684.76 | +86.2% | 50.71 | 43.96 | +15.4% |
| 4,096 | 2854.97 | 1404.62 | +103.3% | 50.21 | 43.26 | +16.1% |
| 8,192 | 2585.95 | 1251.89 | +106.6% | 49.95 | 42.58 | +17.3% |
| 12,288 | 2391.58 | 1115.27 | +114.4% | 49.32 | 41.94 | +17.6% |
| 16,384 | 2263.70 | 1022.60 | +121.4% | 47.99 | 41.42 | +15.9% |
| 32,768 | 1777.45 | 746.17 | +138.2% | 46.00 | 38.57 | +19.3% |
| 65,536 | 1244.05 | 472.56 | +163.3% | 42.49 | 35.12 | +21.0% |
| 131,072 | 778.06 | 281.37 | +176.5% | 36.67 | 29.42 | +24.6% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2456.25 | 1425.52 | +72.3% | 47.02 | 40.94 | +14.9% |
| 4,096 | 2279.82 | 1215.03 | +87.6% | 46.58 | 40.33 | +15.5% |
| 8,192 | 2107.72 | 1103.41 | +91.0% | 46.34 | 39.75 | +16.6% |
| 12,288 | 1913.44 | 996.77 | +92.0% | 45.71 | 39.19 | +16.6% |
| 16,384 | 1896.68 | 918.82 | +106.4% | 44.65 | 38.75 | +15.2% |
| 32,768 | 1551.74 | 679.89 | +128.2% | 42.94 | 36.64 | +17.2% |
| 65,536 | 1081.77 | 447.13 | +141.9% | 39.83 | 33.19 | +20.0% |
| 131,072 | 725.48 | 268.45 | +170.2% | 34.64 | 27.98 | +23.8% |
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
| 0 | 3116.37 | 1630.60 | +91.1% | 78.10 | 57.61 | +35.6% | 173.48 | 62.06 | +179.5% |
| 4,096 | 2840.72 | 1351.13 | +110.2% | 77.31 | 58.64 | +31.8% | 150.60 | 65.36 | +130.4% |
| 8,192 | 2557.58 | 1199.20 | +113.3% | 76.46 | 56.85 | +34.5% | 130.64 | 39.35 | +232.0% |
| 12,288 | 2364.33 | 1051.25 | +124.9% | 80.37 | 56.72 | +41.7% | 173.51 | 45.61 | +280.4% |
| 16,384 | 2272.27 | 957.77 | +137.2% | 67.03 | 49.12 | +36.5% | 139.11 | 57.46 | +142.1% |
| 32,768 | 1820.97 | 704.89 | +158.3% | 65.50 | 39.98 | +63.8% | 140.94 | 61.27 | +130.0% |
| 65,536 | 1229.15 | 461.26 | +166.5% | 65.17 | 31.41 | +107.5% | 128.00 | 32.73 | +291.1% |
| 131,072 | 771.14 | 276.99 | +178.4% | 48.80 | 20.98 | +132.6% | 95.16 | 22.16 | +329.4% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4.svg)

---

<!-- bench:single-mtp-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2455.26 | 1393.42 | +76.2% | 68.31 | 54.03 | +26.4% | 187.19 | 58.75 | +218.6% |
| 4,096 | 2315.34 | 1183.58 | +95.6% | 69.86 | 54.17 | +29.0% | 139.07 | 63.89 | +117.7% |
| 8,192 | 2125.04 | 1043.91 | +103.6% | 77.42 | 50.06 | +54.7% | 140.27 | 39.41 | +255.9% |
| 12,288 | 1905.63 | 943.21 | +102.0% | 71.80 | 54.67 | +31.3% | 159.58 | 58.98 | +170.6% |
| 16,384 | 1906.93 | 864.65 | +120.5% | 72.75 | 45.14 | +61.2% | 128.14 | 52.93 | +142.1% |
| 32,768 | 1573.12 | 651.38 | +141.5% | 57.91 | 37.61 | +54.0% | 90.02 | 60.24 | +49.4% |
| 65,536 | 1090.33 | 435.03 | +150.6% | 55.93 | 33.29 | +68.0% | 116.45 | 31.08 | +274.7% |
| 131,072 | 717.81 | 265.20 | +170.7% | 43.41 | 22.62 | +91.9% | 90.53 | 15.26 | +493.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q6.svg)

## Single user, sampled MTP

Temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05; a story-writing turn
after the cached prefix. Mean of three requests with seeds 1–3.

<!-- bench:single-mtp-sampled-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3084.25 ± 37.17 | 1604.71 ± 6.23 | +92.2% | 67.27 ± 1.75 | 46.41 ± 1.54 | +44.9% |
| 4,096 | 2875.55 ± 52.88 | 1316.00 ± 23.66 | +118.5% | 65.47 ± 3.33 | 48.54 ± 6.25 | +34.9% |
| 16,384 | 2188.52 ± 20.62 | 944.93 ± 7.16 | +131.6% | 58.30 ± 3.25 | 40.69 ± 1.90 | +43.3% |
| 32,768 | 1757.78 ± 24.29 | 699.03 ± 1.46 | +151.5% | 52.89 ± 0.74 | 36.53 ± 1.29 | +44.8% |
| 65,536 | 1230.37 ± 6.12 | 462.90 ± 3.94 | +165.8% | 47.94 ± 0.71 | 26.78 ± 1.57 | +79.0% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q4.svg)

---

<!-- bench:single-mtp-sampled-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2440.66 ± 4.83 | 1383.68 ± 1.03 | +76.4% | 57.41 ± 2.95 | 45.17 ± 3.99 | +27.1% |
| 4,096 | 2323.18 ± 47.07 | 1148.34 ± 14.95 | +102.3% | 61.40 ± 2.17 | 40.17 ± 3.04 | +52.9% |
| 16,384 | 1838.10 ± 37.18 | 852.45 ± 5.06 | +115.6% | 57.50 ± 3.37 | 37.68 ± 2.69 | +52.6% |
| 32,768 | 1506.57 ± 43.22 | 646.04 ± 2.91 | +133.2% | 49.56 ± 0.97 | 33.73 ± 1.83 | +46.9% |
| 65,536 | 1097.57 ± 14.93 | 435.47 ± 0.48 | +152.0% | 46.55 ± 2.41 | 25.03 ± 1.49 | +86.0% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q6.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 50.67 | 43.99 | +15.2% |
| 2 | 91.61 | 71.85 | +27.5% |
| 4 | 148.31 | 106.17 | +39.7% |
| 6 | 179.76 | 129.53 | +38.8% |
| 8 | 209.79 | 146.37 | +43.3% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 46.98 | 40.76 | +15.3% |
| 2 | 85.94 | 66.71 | +28.8% |
| 4 | 142.53 | 95.56 | +49.2% |
| 6 | 175.36 | 122.78 | +42.8% |
| 8 | 204.45 | 133.78 | +52.8% |
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
| 1 | 77.41 | 56.96 | +35.9% | 173.09 | 61.64 | +180.8% |
| 2 | 112.75 | 89.93 | +25.4% | 241.13 | 102.28 | +135.8% |
| 4 | 152.43 | 126.24 | +20.7% | 231.70 | 140.44 | +65.0% |
| 6 | 186.39 | 160.35 | +16.2% | 197.49 | 171.28 | +15.3% |
| 8 | 206.58 | 184.92 | +11.7% | 222.90 | 196.02 | +13.7% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4.svg)

---

<!-- bench:multi-mtp-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 68.17 | 51.22 | +33.1% | 186.57 | 57.94 | +222.0% |
| 2 | 92.71 | 81.47 | +13.8% | 217.05 | 90.28 | +140.4% |
| 4 | 126.16 | 113.95 | +10.7% | 216.98 | 131.20 | +65.4% |
| 6 | 174.93 | 152.44 | +14.8% | 192.59 | 167.68 | +14.9% |
| 8 | 195.29 | 170.86 | +14.3% | 215.30 | 197.08 | +9.2% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q6.svg)

## Loading time

C1, context capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 26B-A4B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 3.82 | 5.63 | +47.4% |
| UD-Q6_K_XL | 4.91 | 6.84 | +39.3% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Context capacity 133121, autoregressive. llama.cpp preallocates its KV
cache, so its footprint does not grow with the prefix.

<!-- bench:memory-q4 -->
| Gemma 4 26B-A4B UD-Q4_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 22.83 | 23.64 | +3.5% |
| 16K prefix, pp4096 + tg128 | 23.57 | 24.18 | +2.6% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4.svg)

---

<!-- bench:memory-q6 -->
| Gemma 4 26B-A4B UD-Q6_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 28.67 | 29.49 | +2.9% |
| 16K prefix, pp4096 + tg128 | 29.41 | 30.03 | +2.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q6.svg)
