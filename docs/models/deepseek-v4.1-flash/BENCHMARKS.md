# DeepSeek V4.1 Flash benchmarks

Provisional numbers from `ds41_load_probe` on the `gpu-test` build until
`gufo bench` supports `deepseek41`. IQ1_M resident (ISTA-DASLab GSQ-RCO) with
the IQ4_NL Engram tables read from disk; one Strix Halo (gfx1151).

| Workload | Speed |
|---|---:|
| Prefill, 2048-token batch at depth 0 | 313 tok/s |
| Prefill, 952-token batch at depth 2048 | 278 tok/s |
| Decode at 2K context | 47.2 ms/token (21.2 tok/s) |

Prefill timings include logits for every position (the `--score` run).

`gufo serve` (release build, 16K context): a 9,160-token chat prompt
prefilled at 307 tok/s in six chunks of up to 2,048 tokens, then decoded at
19.8 tok/s (before the FP32-activation GEMV); repeating it reused all 9,160 prompt tokens from the cache.
Benchmarks at depths 8K/32K/131K follow once `gufo bench` serves the model.
