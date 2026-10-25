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
| pp2048 | 394.25 ± 4.17 |
| tg64 | 23.93 ± 0.04 |
| pp2048 @ d8192 | 396.69 ± 0.91 |
| tg64 @ d8192 | 23.41 ± 0.01 |
| pp2048 @ d32768 | 361.74 ± 0.78 |
| tg64 @ d32768 | 22.89 ± 0.01 |

Prefill batches no longer shrink with the session context: the same
2,048-token prompt (`ds41_load_probe --batch`) prefills at 338 tok/s in a
131K-token context (was 245, 409-token batches) and at 342 tok/s in the
native 1M context (was ~120, 51-token batches) with the compressed rows
stored as bfloat16.

`gufo serve` (16K context, before the FP32-activation GEMV): a 9,160-token
chat prompt prefilled at 307 tok/s in six chunks of up to 2,048 tokens, then
decoded at 19.8 tok/s; repeating it reused all 9,160 prompt tokens from the
cache.

Depth 131K is not measured yet.
