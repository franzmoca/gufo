# DeepSeek V4.1 Flash benchmarks

`gufo bench` on the release build: IQ1_M resident (ISTA-DASLab GSQ-RCO,
112.43 GiB) with the IQ4_NL Engram tables read from disk; one Strix Halo
(gfx1151). Two repetitions per row; greedy decoding.

```sh
gufo bench -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf \
  -p 2048 -n 64 -d 0,8192,32768 -r 2
```

| Test | t/s |
|---|---:|
| pp2048 | 352.06 ± 0.94 |
| tg64 | 21.36 ± 0.13 |
| pp2048 @ d8192 | 348.61 ± 2.84 |
| tg64 @ d8192 | 20.32 ± 0.49 |
| pp2048 @ d32768 | 268.26 ± 1.92 |
| tg64 @ d32768 | 19.52 ± 0.26 |

`gufo serve` (16K context, before the FP32-activation GEMV): a 9,160-token
chat prompt prefilled at 307 tok/s in six chunks of up to 2,048 tokens, then
decoded at 19.8 tok/s; repeating it reused all 9,160 prompt tokens from the
cache.

Depth 131K is not measured yet.
