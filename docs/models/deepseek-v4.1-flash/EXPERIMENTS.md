# DeepSeek V4.1 Flash experiments

IQ1_M resident on one Strix Halo (gfx1151). Speeds come from
`ds41_load_probe --batch` on the `gpu-test` build; prefill is a 2K-3K token
prompt, decode is greedy at 2K context. Quality gates: the 13-token smoke
continuation (same greedy tokens, KL against the oracle) and the 3K-token
teacher-forced score against the oracle's perplexity 6.309
([QUALITY.md](QUALITY.md)).

| Experiment | Decision / evidence |
| --- | --- |
| Layer-major batched prefill (512-token batches, MMQ for every quantized projection, IQ1_M added to the MMQ tiles) | Retained. 3K perplexity 6.300, argmax agreement 96.2%. Prefill 104.7 tok/s at depth 0 falling to 60.2 tok/s at 2.5K. Token-at-a-time prefill was ~13 tok/s. |
| WMMA prefill attention: 32 heads per block, half Q/K tiles, fp32 online softmax over window and indexed compressed rows | Retained. Attention 9.0 s → 0.55 s per 2K prefill + 16 decode steps; prefill 104.7 → 140.9 tok/s at depth 0, 60.2 → 89.5 at 2.5K. Perplexity 6.308. |
| Routed MoE launch bounds from the real expert buckets (largest bucket bounds the column-tile grid, tile width from the bucket distribution) | Retained. IQ1_S expert GEMM 6.1 → 2.9 s; prefill 140.9 → 198.9 tok/s at depth 0. Bounded and unbounded launches agree with the matvec path to the same 2e-4 rounding (`ds41.mmq_typed`). |
| WMMA indexer scores: 16 tokens × 128 compressed rows per block, half operands, fp32 relu-weighted head sum | Retained. Prefill no longer slows with depth: 200 tok/s flat to 3K (was 198.9 → 115.8). Perplexity 6.296. |
| Split-row decode attention: 20 row splits × 16 heads per block, rows staged once in LDS, fp32 online softmax, merge with the sink | Retained. Matches an fp64 reference to 1.6e-7 (`ds41.attention`). Decode at 2K 89 → 69.8 ms/token; decode-path 3K perplexity 6.307. |
| Sinkhorn with the hc = 4 matrix as a template constant | Retained. The runtime-sized matrix lived in scratch memory: 90 → 2.5 µs per call, decode 69.8 → 62.0 ms/token. |
| Non-finite guard folded into the matvec store, dropping the per-call clear and sanitize launches | Retained. Same values; 723 → 443 small launches per token; decode 62.0 → 58.7 ms/token. |
| Eight output-LoRA groups in one channel-batched matvec | Retained. Bit-identical at one token (7e-8 at three, from a different row tiling). Decode 58.7 → 56.3 ms/token. |
| Engram rows read on worker threads from the start of each evaluation, overlapping the layers before 1 and 14 | Retained. Bit-identical; decode 56.3 → 53.7 ms/token. |
| BF16 matmul with 16-byte weight loads (eight weights per lane per step) | Retained. Decode 53.7 → 50.5 ms/token, prefill 200 → 233 tok/s. Smoke KL unchanged in range. |
| Prefill batches of 1024 / 2048 tokens (from 512) | 2048 retained: 233 → 261 → 292 tok/s; experts average 32 rows per batch instead of 8. Activations take ~1 MiB per token; 6.8 GiB stay free after the decoder allocates. The context budget still lowers the cap at long contexts. |
| Batched hc mixes: two tokens and all 24 rows per block (state read once) | Retained for batches of 16+ tokens. Prefill 292 → 313 tok/s; perplexity 6.310. Decode keeps the per-row kernel (one block per token would idle the GPU). |

## Measured limits

- The dense Q5_K decode matvecs read about 4.8 GB per token in 23.6 ms
  (~200 GB/s), close to the device's practical bandwidth. Decode now spends
  about 47 ms of GPU time per token at 2K.
