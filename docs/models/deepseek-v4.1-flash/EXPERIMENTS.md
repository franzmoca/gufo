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
| Single-token Q5_K/Q8_0 projections through an FP32-activation GEMV forked from the Gemma 4 K-quant GEMV (no Q8_1 activation pass), eight waves split into 8 / S row groups of S slices | Retained. DRAM-cold bandwidth, mmvq → GEMV: q_b 190 → 221, wo_b 191 → 219, grouped wo_a 200 → 219, shared gate 179 → 210, shared down 172 → 207, wq_a 154 → 198, wkv 102 → 166 GB/s (measured DRAM read peak 235 GB/s). S = 2 for K ≤ 2304, 4 for K ≤ 5120 with M ≤ 4096, else 8. Matches an fp64 dot to 2e-7 (mmvq: ~5e-3 from Q8_1 activations). Decode 50.5 → 47.2 ms/token; decode-path 3K perplexity 6.303, agreement 96.1%, mean |ΔNLL| 0.077. |
| Single-token routed experts through an FP32-activation expert GEMV (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S): IQ1 groups as d s (sum n x + (delta - 1) sum x), IQ1 codebook in LDS, IQ1_S loads two blocks ahead | Retained for IQ1 projections and IQ2 down projections; IQ2_XXS gate/up stay on mmvq (141 → 154 µs). DRAM-cold six-expert launches, mmvq → GEMV: IQ1_S gate/up 87.5 → 79.3 µs, down 94.1 → 85.2; IQ1_M 91.3 → 89.9, 106.1 → 95.9; IQ2_XXS down 208.5 → 147.2; IQ2_S down 224.5 → 158.7. Matches mmvq to its Q8_1 rounding (5e-3 to 9e-3, `ds41.mmq_typed`). Decode at 2K 47.1 / 47.8 → 46.5 / 46.8 ms/token (alternating runs); decode-path 3K perplexity 6.305, agreement 96.5%, mean \|ΔNLL\| 0.075. |
| Expert GEMV variants (same DRAM-cold launches) | Rejected, each slower than the retained kernel: mmvq with 2, 4 or 8 waves per row (IQ1_S gate/up 91-118 µs); activations held in registers across 2-8 rows per lane (110-150 µs); activations staged in LDS, with or without bank padding (100-300 µs); non-temporal weight loads (102 µs); 32-row tiles staged into LDS with dword loads and one row per lane (126-245 µs, register spills); per-weight decode to floats before the FMA (95-100 µs). |
| MMQ scratch pool keeps freed buffers for reuse on the freeing stream (was `hipMallocAsync`/`hipFreeAsync` per call: ~0.45 ms of host time per free, ~0.9 ms for the first allocation after each synchronize) | Retained. Same kernels and values (`ds41.mmq_typed`). `gufo bench` -p 512 -n 64 -r 3: tg64 21.83 → 22.17 t/s, tg64 @ d2048 20.96 ± 0.65 → 21.88 ± 0.01, pp512 303.6 → 308.3. |
| Engram rows of batches up to 16 tokens staged in pinned memory and uploaded asynchronously (the pageable upload waited for layer 0 to drain), four rows per read thread (was 24, so one thread per decode token) | Retained. Same generated tokens. Traced 2K decode: the ~0.5 ms stall before layer 1 is gone, step 48 → 46.7 ms with the pool change. `gufo bench` tg64 22.16 → 22.20, tg64 @ d2048 21.88 → 21.95 t/s. |
| hc collapse fused with the following RMSNorm (block of 1,024 threads per token, values kept in registers; the separate 256-thread norm was latency-bound at 10 µs per 5,120-value row), and the rope tail fused with the FP8 fake quantization of the KV latent and the attention output | Retained. 160 fewer launches per token. Rope + FP8 is bit-identical; the fused norm changes the sum order: decode-path 3K perplexity 6.291, agreement 96.0%, mean \|ΔNLL\| 0.073; batched 6.324, 95.9%, 0.078; smoke continuation keeps all 9 greedy tokens, KL ≤ 3.2e-3 per step. `gufo bench` -p 512 -n 64 -r 3: tg64 22.29 → 22.81, tg64 @ d2048 21.94 → 22.43 t/s, pp512 308.0 → 316.7, pp512 @ d2048 323.6 → 331.2. |

## Measured limits

- Measured DRAM read peak (`tools/bench/gfx1151_peak.hip`): 235 GB/s. The
  dense decode projections run at 207-221 GB/s on the large shapes; decode
  spends 42.6 ms of GPU time per token at 2K (47.2 ms wall).
- Routed expert matvecs: a plain read of the same six DRAM-cold 2.3 MB
  experts streams at 230-238 GB/s (58 µs); the expert GEMV reaches 160-175
  GB/s on IQ1 (79-96 µs) and mmvq 145-170 GB/s. Both are bound by the
  texture addresser (`TA_BUSY` ~92% of active cycles): every 32-value
  sub-block costs four 2-byte weight loads and eight activation loads per
  lane.
