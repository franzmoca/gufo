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
| pp2048 | 703.07 ± 10.04 |
| tg64 | 23.88 ± 0.13 |
| pp2048 @ d8192 | 677.76 ± 1.79 |
| tg64 @ d8192 | 23.44 ± 0.02 |
| pp2048 @ d32768 | 585.30 ± 1.84 |
| tg64 @ d32768 | 22.92 ± 0.01 |

Prefill runs the routed experts, the dense Q5_K projections and attention on
the FP16 matrix cores (previous card: pp2048 421.80, @ d8192 414.37,
@ d32768 378.66; decode unchanged).

Prefill batches no longer shrink with the session context: the same
2,048-token prompt (`ds41_load_probe --batch`, first pass over the weights)
prefills at 384 tok/s with a prompt-sized context, 386 tok/s in a
131K-token context (was 245 with 409-token batches) and 381 tok/s in the
native 1M context (was ~120 with 51-token batches).

## Memory

GTT in use (`ds41_load_probe --upload` and `gufo serve`, native 1M context):

| State | Before | Now |
|---|---:|---:|
| Weights loaded | 112.69 GiB | 112.27 GiB |
| One session | 118.32 GiB | 114.55 GiB |
| Two sessions (`-j 2`) | does not fit | 115.45 GiB |

With two sessions, 6.3 GiB of host memory stays available. A 2,048-token
prefill peaks 0.3 GiB above the idle session.

`gufo serve` (16K context, before the FP32-activation GEMV): a 9,160-token
chat prompt prefilled at 307 tok/s in six chunks of up to 2,048 tokens, then
decoded at 19.8 tok/s; repeating it reused all 9,160 prompt tokens from the
cache.

Depth 131K is not measured yet.
