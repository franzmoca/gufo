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
| hc mixes and Sinkhorn as one kernel below 16 tokens: each mix row split over 8 blocks with 16-byte loads, the last block of a token reduces the partial sums and runs its Sinkhorn | Retained. 80 fewer launches per token; the mix itself stays at ~14 µs (2 MB of F32 weights; 1, 4 and 8 blocks per row measured 14.1, 14.1 and 13.7 µs). Decode-path 3K perplexity 6.310, agreement 96.0%, mean \|ΔNLL\| 0.073; smoke continuation keeps all 9 greedy tokens, KL ≤ 5.3e-3. `gufo bench` tg64 22.75 → 22.95, tg64 @ d2048 22.43 → 22.54 t/s. Batched prompts keep the separate Sinkhorn kernel in its original form: routing it through the shared device function changed fast-math rounding and moved the batched 3K scores (6.324 → 6.321, agreement 95.9% → 95.4%, mean \|ΔNLL\| 0.078 → 0.082); with the original form they are bit-identical again. |
| Indexer top-k threshold by 8-bit radix select (four LDS histogram passes, the digit found in one wave) instead of a 32-step binary search with a block scan per step | Retained. Same threshold, so bit-identical (decode-path 3K scores unchanged); `ds41.candidate_mask` adds top-k cases with tied scores. Per call at 2K decode 45.3 → 6.9 µs; largest prefill call 2.5 → 0.34 ms. `gufo bench` tg64 @ d2048 22.55 → 22.74, tg64 @ d32768 20.91 → 21.55, pp512 @ d32768 257.0 → 298.0 t/s. |
| Indexer scores for up to 4 tokens in fp32 without matrix cores (the 16-token WMMA tile ran one live token and a block barrier per head): 64 keys per block staged by coalesced loads into padded shared memory, a thread per (key, 8 heads), groups summed in shared memory | Retained. Per call at 18.5K decode 195 → 74 µs (a thread per key with strided key loads: 112 µs). Matches fp64 to 2.4e-7 (matrix-core path 3.7e-4; `ds41.attention`). Decode-path 3K perplexity 6.311, agreement 96.0%, mean \|ΔNLL\| 0.073. `gufo bench` tg64 @ d2048 22.73 → 22.84, tg64 @ d32768 21.56 → 22.17 t/s. |
| Single-token gate and up projections with SwiGLU in one launch: a wave takes the same rows of both matrices (IQ1 routed experts, with the route weight; Q5_K shared expert) | Retained. Bit-identical (decode-path 3K scores unchanged; `ds41.mmq_typed` compares fused and unfused within 4.3e-8). 120 fewer launches per token; at 2K the routed pass is 144 µs against 153.5 µs for gate, up and SwiGLU, the shared pass 73.0 against 75.3 µs. `gufo bench` tg64 23.31 → 23.65, tg64 @ d2048 22.84 → 23.17 t/s. |
| Single-token q_a and KV projections (both Q8_0 over the same input) as one 1,792-row GEMV launch, and both RMSNorms plus the KV rope + FP8 quantization in one kernel | Retained. Bit-identical on the decode and batched paths once the norm keeps rms_norm's loop form (a register-array form rounded differently under fast-math: max 9.5e-7, batched agreement 95.4%); `ds41.attention` compares it with the separate launches. 120 fewer launches per token. `gufo bench` tg64 23.67 → 23.82, tg64 @ d2048 23.18 → 23.43 t/s. |
| Routed-expert sum (six down rows plus the shared expert) folded into the FFN hc expand | Retained. Bit-identical; 40 fewer launches. `gufo bench` tg64 23.81 → 23.88, tg64 @ d2048 23.43 → 23.47 t/s. |
| Attention output rope + FP8 quantization fused into the split-attention merge (head row staged in shared memory) | Rejected. 40 fewer launches but no measured gain (`gufo bench` tg64 @ d2048 23.47 → 23.48 ± 0.01 t/s), and the merge rounds differently in rare cases (3K decode path: 115 argmax changes, perplexity 6.311 → 6.309). |
| IQ2_XXS routed gate/up through the fused FP32 pair instead of two mmvq launches and SwiGLU | Retained. 252 µs against 251 µs for the three launches at 2K, so the gain is the launches (`gufo bench` tg64 23.84 → 23.94, tg64 @ d2048 23.46 → 23.57 t/s). Decode-path 3K perplexity 6.313, agreement 96.0%, mean \|ΔNLL\| 0.073. |
| Split decode attention with 10 or 40 row splits (20 retained) | Rejected. At 2K, split + merge per token: 20 splits 1,107 + 208 µs; 10 splits 1,312 + 94 µs (half the partial traffic, half the blocks); 40 splits 1,213 + 438 µs. |
| Routing kernel with one wave or four waves per token (eight retained) | Rejected. Per call at 2K decode: 8 waves 7.0 µs, 4 waves 7.4 µs, 1 wave 13.4 µs (12 serial softplus evaluations per lane). Same selections. |
| Prefill batch sized from free memory instead of context: indexer scores, candidate mask and top-k run in groups of up to 256 MiB of score rows, the candidate mask keeps one byte per 8-row block (was one per row, ~2 GiB at 1M context for 2048 tokens) | Retained. The batch had shrunk with context to stay under 256 MiB of score rows: 1,538 tokens at 35K, 409 at 131K, 51 at the native 1M. The same 2K prompt prefills at 344 tok/s with a prompt-sized context; at 131K 245 → 338 tok/s (2048-token batch), at 1M ~120 → ~255 tok/s (576-token batch, 1.7 GiB RAM left). `gufo bench` pp2048 @ d32768 330.5 → 349.3 t/s. Bit-identical batched 3K scores at a prompt-sized context; 18.5K scored at 1M context: perplexity 6.273, agreement 95.9% (97.2% in the masked 16,383-17,399 span). |
| Prefill glue: every MMQ type but Q8_0 zeroes non-finite outputs at the tiled store (was a whole-buffer sanitize pass after 642 of 722 launches per 2K prefill, 150 ms), and batched routed gate/up of one type run as one pair that quantizes the input once and stores SwiGLU × route weight from the up pass (was two routed launches, each quantizing and gathering the input, then a SwiGLU pass) | Retained. The store-side guard is bit-identical (batched 3K scores); the pair gives the same gate rows and SwiGLU within one ulp (4-5e-8, `ds41.mmq_typed` for all six types), which shifts the batched 3K run to perplexity 6.300 (was 6.324), agreement 96.1%, mean \|ΔNLL\| 0.081. `gufo bench` pp2048 380.2 → 395.5 t/s; tg unchanged. |
| Prefill batches of 4096 tokens (experts average 64 rows) | Rejected for now. 16K-token prompt 345.4 → 361.5 tok/s (+4.7%) for 2 GiB more resident activations (4.2 GiB RAM left at 16K context). Doubling the rows per expert cut the routed MMQ cost per token only 10-20% (IQ1_S 10.2 → 9.3 µs per token and launch). |
| Routed MMQ column-tile width forced to 16, 32, 48 or 64 (the bucket model picks 32 at 2048 tokens) | Rejected; 32 retained. Per 2048-token batch of a 4K prompt: 16 → 6.24 s, 32 → 5.80 s, 48 → 6.03 s, 64 → 6.20 s. Tile fill (real / issued columns) 82%, 67%, 56%, 48%: narrow tiles pad less but repeat the weight-panel unpack, about 40% of a 32-column tile's cost. The largest bucket (up to 1,537 of 2,048 tokens) sizes the column grid for every expert, but launch time is flat in it (11 → 50 column tiles: 20.5 → 21-23 ms), so empty blocks cost little. |
| Compressed KV and index-K rows stored as bfloat16 (FP4 values times an E4M3 or power-of-two scale have at most 6 significant bits, so the conversion is exact); the FP4 quantization writes the state directly instead of quantizing in place and copying | Retained. Bit-identical on the batched and decode 3K paths. State at the native 1M context ~6.3 → ~3.1 GiB: the 2K prompt there runs as one 2048-token batch at 342 tok/s (was 576-token batches, ~255), 3.1 GiB RAM left. Saved states change format (magic `S4STav2`). |
| Batched output LoRA as one channel-batched MMQ (8 groups as channels: one input quantization, one launch; was per group a row copy in, a quantization, a launch and a row copy out: 640 copies, 128 ms per 2K prefill), and q_a/KV and shared gate/up each quantizing their common input once (dense pair) | Retained. Bit-identical batched 3K scores; `ds41.mmq_typed` checks the grouped and pair launches against separate launches exactly for all six types. `gufo bench` pp2048 385.6 → 409.8 t/s against the binary before the bfloat16 state in the same session (395.5 in the previous session). |
| Routed MMQ tile width per expert: 16, 32 or 64 columns from each bucket's size (cost ceil(c / w) × (w + 21) column-equivalents, the panel term fitted from the fixed-width sweep), one launch per width over its experts only (the kernel skips other experts' blocks) | Retained. Prefill 2048-token batch 5.84 → 5.42 s; 18.5K-token prefill 56.9 → 50.3 s (with the channel-batched output LoRA). Tile widths change the matrix-core accumulation split: class-split launches match one launch to ≤1.3e-7 (`ds41.mmq_typed`, all six types, every class used); batched 3K perplexity 6.307 (oracle 6.309), agreement 95.5%; 18.5K perplexity 6.277, agreement 95.9% (was 6.272, 95.9%). Adding 128-column tiles or moving the panel term within 14-28: within noise. `gufo bench` pp2048 409.8 → 421.8, pp2048 @ d32768 → 378.7 t/s. |
| Token embedding left in the mapped file instead of device memory (it is dequantized on the host one row per token) | Retained. Resident weights 112.69 → 112.27 GiB GTT. |
| One batch-scratch set shared by every decoder of an engine and context (the engine serializes evaluations and no scratch value outlives one), and the host embedding rows uploaded once per token then copied on the device into the four hc streams (was four host copies) | Retained. Bit-identical on the batched and decode 3K paths. A second `gufo serve` session at the native 1M context costs its state only (0.90 GiB, was 5.6 GiB with its own scratch); host staging 160 → 40 MiB. |
| Compressed KV and index-K rows packed as FP4: E2M1 codes two per byte plus one scale byte per block (E4M3 for KV rows, block 16; the exponent of the power-of-two scale for index-K rows, block 32); the readers decode on load, eight values per code word in the batched attention and indexer staging | Retained. Bit-identical on the batched 3K and 18.5K and decode 3K paths (decoding multiplies the same E2M1 value and scale as the quantization; `ds41.attention` checks the packed bytes against the fake-quantized floats bit for bit). State at 1M ~3.1 → 0.90 GiB; saved states and cache snapshots shrink by the same factor (magic `S4STav3`). Four-value loads cost 7% in the batched attention kernel at 32K; the eight-value staging brings it to within 1% of bfloat16 (10.96 against 10.85 s of a 32K-depth bench trace), the indexer to parity. |
| Batch scratch overlaid by phase: the attention, FFN and Engram buffers, each dead outside its phase, share one region | Retained. Bit-identical on the batched and decode 3K paths. Scratch per 2048-token batch 2.1 → 1.2 GiB. |
| Candidate-block mask as one bit per block (was one byte) | Retained. Bit-identical 18.5K scores (the mask acts above 16K). 256 → 32 MiB at 1M. |
| Batched BF16 matmuls (router, compressor, indexer projections; 16 tokens and up) on the matrix cores: FP32 activations split into three exact bfloat16 terms, accumulator restarted every 32 values and summed in FP32 registers | Retained. 2K prefill BF16 time 292 → 103 ms, `gufo bench` pp2048 414.6 → 433.0 t/s (one run each). The matrix-core accumulator loses precision over long chains (2e-5 relative over k = 5120 with one accumulator; restarts every 16, 64, 128 or 256 values measured 1.1e-6, 7.5e-7, 1.1e-6 and 2.1e-6), so the result is within 8e-7 of the largest output against 2.5e-7 for the FP32 kernel (`ds41.attention`). Decode keeps the FP32 kernel (bit-identical). Batched 3K perplexity 6.307 → 6.326, agreement 95.5 → 95.1% (exact-arithmetic reorderings earlier measured 6.300-6.324, 95.5-96.1%); 18.5K 6.277 → 6.279, agreement 95.9 → 95.8% (oracle 6.267). |
| Component cache rows in chunks of 2048 rows (from 256; rows are 4-256 bytes) | Retained. Lending ~400 small chunks per 10K-token slot spilled at ~1.7 GB/s; with 2048-row chunks 8 MB spill in 0.8 ms. Restore queue time in cache-shared-prefix 16.7-17.3 → 13.7-16.0 ms. |
| One private component for all 40 window rings and one for the 4 compressor frontiers (from one per layer: 63 → 21 components) | Retained. Against the whole-state snapshot, per-request capture time median 0.107 → 0.066, restore 0.71 → 0.44. |
| No decoder reset when the cache invalidates a slot (`Sync` resets an invalid session, a restore replaces every part) | Rejected. Queue times unchanged (cache-shared-prefix `long_tasks_1` 16.7 → 16.8 ms). |
| DSpark draft width fixed at 1-5 (Lucebox Q2_K/Q4_K drafter; roadmap prompt, 1,871 tokens, 256 greedy tokens; plain 23.46 tok/s) | Basis for the cost model. Width 1: 25.28 tok/s, 77.1% accepted, 1.78 tokens/cycle; 2: 28.16, 68.8%, 2.37; 3: 27.88, 56.1%, 2.69; 4: 25.90, 45.8%, 2.84; 5: 23.12, 37.4%, 2.88. A draft costs 12.5 ms; verifying 2-6 rows costs most of a cycle, so long blocks lose once acceptance drops. Conditional acceptance per position: about 0.77, 0.78, 0.53, 0.47, 0.27. |
| DSpark draft width chosen per cycle: running estimates of each position's conditional acceptance and of each width's cycle time, width maximizing expected tokens per millisecond, plain steps when no width beats one | Retained. Same run 28.49 tok/s (+21%), 104 cycles, 2.46 tokens/cycle. Variants: a 32-step plain lockout after a losing estimate 26.0 tok/s; one-draft probes with backoff 4 → 64 plain steps 27.45; also extending the width after a fully accepted block over-explored (rejected); relaxing undrafted positions toward the prior 28.49 (retained). Acceptance estimates restart with each request (a greedy request after a sampled one otherwise drafted less); cycle times carry over. |
| DSpark drafters: unsloth Q8_0 (Q8_0 dense, MXFP4 experts, 7.42 GiB) against Lucebox Q2_K/Q4_K (4.71 GiB) | Q2_K/Q4_K retained as the recommended drafter. Roadmap prompt, 128 tokens: Q8_0 26.49 tok/s, 51.7% accepted; Q2_K/Q4_K 27.57, 52.6%. Chat coding prompt (45 tokens, 384 tokens): Q8_0 39.17 tok/s, 76.3%, 4.09 tokens/cycle; Q2_K/Q4_K 37.85, 76.3%, 3.84 (plain 23.5). Host memory after load: 2.1 GiB with Q8_0, 4.7 GiB with Q2_K/Q4_K. |
| Multi-token decode GEMVs (2-8 verification rows) load the next Q5_K block's raw words before decoding the current one, keeping each column's dot order | Retained. Bit-identical (`ds41.mmq_typed`). `--spec-bench`, 1-8 rows at 137-186 tokens: 41.8 / 55.8 / 67.3 / 77.7 / 88.5 / 99.0 / 112.0 / 113.9 → 41.8 / 54.9 / 66.0 / 76.4 / 88.1 / 97.1 / 109.6 / 110.8 ms. |
| The expert-group plan built in O(slots²) (each slot finds its group's first slot, leaders count earlier leaders) instead of O(slots³) | Retained. Same plan (`ds41.mmq_typed`); 40 plan launches per 6-row pass 1.18 → 0.11 ms; `--spec-bench` 6 / 7 / 8 rows 97.1 → 96.1, 109.6 → 107.7, 110.8 → 109.3 ms. |
| Prompt lookup in DSpark sessions: a cycle whose context repeats earlier tokens (PromptLookup gates: 12 matching tokens, 24 and prompt-only for sampled requests, copies only as far as competing occurrences agree) verifies up to six copied tokens instead of drafting | Retained. File edit, 1,302 tokens, 512 generated: 43.88 → 56.43 tok/s, 71 of 78 cycles copied, 414 of 420 copies accepted; roadmap 28.95 and chat 37.81 tok/s (0 and 3 lookup cycles). Seven copies (eight rows) changed the output: the batched attention kernel starts at eight rows. |
| Decoder SWA bounded replay: prompts run layers 20-39 over their last 140 positions (DeepInfer's V4.1 default), layer 20 still compresses every position, and the replayed windows start at the first replayed position | Retained. Prefill 1.77-1.97× faster (8,863 tokens 14.70 → 7.75 s; 23,848 tokens 40.37 → 22.84 s); continuation NLL within 0.008 of exact prefill and top-1 agreement 254-255 of 256 ([QUALITY.md](QUALITY.md)). Serve: 11,327-token prompt 16.95 → 9.52 s. |
| Prompt lookup without a drafter, through the same speculative step | Retained. File edit 23.59 → 54.61 tok/s, chat 23.81 → 24.30 tok/s, greedy output identical. The first version evaluated each emitted token before returning it: serve's `cache` suite showed decode 37-39 → 42 ms per token and the second token 43 → 86 ms after the first, one evaluation more per request than plain decoding (which emits a token, then evaluates it). Steps now emit the token after the accepted copies unevaluated and verify it as the next step's first row: against the previous build, all 479 per-request timings pass or stay inconclusive (single flags that did not repeat over three alternating pairs), and the 30 completed replies are identical. With the DSpark drafter, five alternating pairs left only flags that changed direction between pairs, except one disk restore after a restart that waited 22-88 ms for a transfer stream in three candidate runs and 39 ms in one unchanged-build run (about 2 ms otherwise): inconclusive, and present without this change. |
| Grouped routed-expert kernel for 2-8 verification rows (IQ1_S gate/up is 17.9 ms of a 6-row pass for 21.2 distinct experts per layer; its weights alone would take about 13.4 ms at the single-row kernel's 203 GB/s) | Not adopted, all slower: the next weight block loaded across loop iterations (routed total 46.8 → 48.2 ms); per-group activation sums precomputed once per column (no gain); 10 waves per SIMD (177 → 144 VGPRs) 17.9 → 20.5 ms; two rows per lane sharing each activation load 17.9 → 45.9 ms (register pressure); 2-column tiles in a runtime loop (24 → 10 KB of code, 177 → 161 VGPRs) 17.9 → 19.5 ms. Counters: the 6-row kernel issues 4.3× the VALU instructions of the single-row kernel for 3.5× the experts, at about 55% VALU issue. |
| Vision encoder linears: hipBLAS `GemmEx` (BF16, small-tile kernels), Flash-Next's dense BF16 WMMA GEMM, hipBLASLt | hipBLASLt retained. Carrots encode: hipBLAS 1,460 ms, WMMA GEMM 1,030 ms; with the fused attention, WMMA GEMM 302 ms and hipBLASLt 249 ms. 4000 × 3000 with the fused attention: WMMA GEMM 1,252 ms, hipBLASLt 853 ms. Linears at 29 TFLOPS on the capped image. Spans within BF16 noise of the official forward. |
| Vision attention fused (online softmax, BF16 WMMA, 16 queries per wave) instead of tiled score GEMMs, softmax and value GEMMs | Retained. Carrots 1,030 → 302 ms. |
| Fused vision attention computing S^T = K Q^T and O^T = V^T P^T, so each lane keeps one query's softmax state and the half-waves swap probabilities (`permlanex16`) instead of shuffles and an LDS round trip; branchless BF16 packing; K/V tiles staged in LDS for eight waves | Retained. 4000 × 3000: 853 → 742 ms; carrots 249 → 236 ms. Bit-identical between the transposed variants. Variants: the hardware exponential (same output, 2%, within noise); LDS staging before the transposition (slower, the kernel was ALU-bound with 256 VGPRs); 16 waves (783 ms); register prefetch of the next tile (763 ms). |


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
| Batched IQ1_S routed experts on the FP16 matrix cores (`ds41_routed_f16.hip.cpp`): FP16 activations (one conversion per layer; the gate/up pass stores the SwiGLU rows in FP16 for an IQ1_S down pass), each weight decoded to FP16 with its sub-block scale folded in from an LDS table of the codebook's FP16 high bytes, host-built 64-slot tiles per expert, weight rows as the B operand. Each half-wave decodes one of the two 8-value groups of a K16 step and takes the other with `permlanex16`; half 1 holds the step with its groups swapped and reads its activation fragment swapped to match (each half of the wave uses its own operand copy) | Retained. `ds41_mmq_bench --type 19` on recorded 2K-prefill expert counts: gate/up+SwiGLU 37.5 → 22.8 ms, down 21.4 → 12.2 ms per layer (MMQ). Max error against the FP32-activation expert GEMV 0.18% of the largest SwiGLU row value and 0.05% for down, against 3.5% and 1.2% for MMQ's Q8_1 inputs. 2K batch 5220 → 4680 ms, 952-token batch 2714 → 2348 ms; batched 3K perplexity 6.326 → 6.311 (oracle 6.309), agreement 95.1 → 95.9%. Steps that measured: ordinary instead of nontemporal weight loads 30.5 → 23.8 ms (nontemporal loads refetched the partly used lines; L2 misses 39 → 31 M); FP16 instead of FP32 activation staging 32.2 → 30.5 ms; the group split 23.8 → 22.8 ms. No gain: 16-byte instead of 4-byte weight loads, weights two blocks ahead (32.7 → 32.2 ms), occupancy hints. The 4-byte nibble codebook with the 1024-magic decode instead of the FP16 byte table: 23.8 → 25.7 ms. |
| The FP16 routed kernel for IQ1_M and IQ2_XXS too (a format trait per type: IQ1_M takes its 3-bit scale per 16 values and its delta per group, from the block's spread f16 scale; IQ2_XXS reads each codebook entry as eight FP16 magnitudes and each sign index as eight FP16 sign bits from LDS, then one XOR and one multiply per pair) | Retained. `ds41_mmq_bench` against MMQ, gate/up+SwiGLU and down per layer: IQ1_M 49.8 → 24.2 and 28.0 → 12.7 ms, IQ2_XXS 39.3 → 27.2 and 22.8 → 14.1 ms (IQ2_S stays on MMQ). 2K batch 4680 → 4337 ms (472 tok/s), 952-token batch 2348 → 2154 ms; batched 3K perplexity 6.316, agreement 96.2%; 18.5K prefill 48.5 → 39.6 s, perplexity 6.279 (unchanged, oracle 6.267), agreement 95.8 → 96.0%. |
| Batched dense Q5_K projections (q_b, the grouped output LoRA, o_b, shared expert) on the FP16 matrix cores (`ds41_dense_f16.hip.cpp`): 128 rows x 128 tokens per workgroup, 16 rows per wave against eight 16-token tiles, each lane loading only the half-blocks its half-wave decodes (no LDS for weights), FP32 activations prefetched in registers one 64-value stage ahead and rounded to FP16 into LDS, each half-wave reading eight values of an activation fragment and taking the other eight with `permlanex16` | Retained; replaces MMQ and its Q8_1 passes for 9 tokens and up. `ds41_mmq_bench --dense` at 2048 tokens, MMQ (with its quantization) → FP16: q_b 3.52 → 2.51 ms, output LoRA (8 groups) 4.04 → 3.90 ms, o_b 6.95 → 5.30 ms, shared gate 2.26 → 1.84 ms, shared down 2.00 → 1.44 ms; faster from 16 tokens up. Relative error 3.5e-4 against the FP32-activation GEMV. 2K batch 4337 → 3977 ms (515 tok/s), 952-token batch 2154 → 2102 ms, 18.5K prefill 39.6 → 37.2 s; batched 3K perplexity 6.285 (oracle 6.309), agreement 95.9%; 18.5K 6.269 (oracle 6.267), 96.0%. Steps: FP16 activations from a separate conversion pass were slower than staging FP32 in the kernel; staging loads issued one at a time (no register prefetch) and a branch-free full-tile variant (the compiler hoisted it into 256 VGPRs and 760 spills) both lost; the fragment exchange instead of two 16-byte LDS reads per lane took o_b 7.01 → 6.29 ms (it measured slower in the routed kernel, whose fragments feed two products). |
| Batched attention rewritten (`attention_f16_kernel`): 256 threads per (token, 32 heads), window and compressed rows in separate 16-row tiles prefetched in registers one tile ahead, each thread's two staged columns kept in registers as the B operands of the output products (each half-wave multiplies by its own operand copy, so its own columns and the opposite half's via `permlanex16` cover them), Q fragments in registers, FP4 rows decoded through LDS tables with one FP16 multiply (exact), DPP reductions; the q rotation and the output's inverse rotation and FP8 rounding run inside the kernel from a per-batch (cos, sin) table, replacing the two separate passes | Retained. `ds41_attention_test --bench` at 2048 tokens with the rotations: window-only layers 3.1 ms, ratio-1 layers 9.4 ms, ratio-2 8.5 ms (old kernel plus passes 13.5 ms per layer on average, plus 4.5 ms of rope passes). In the 2K trace attention 540 → 364 ms and the rope passes are gone. 2K batch 3977 → 3562 ms (575 tok/s), 952-token batch 2102 → 1881 ms, 18.5K prefill 37.2 → 33.3 s; batched 3K perplexity 6.294 (oracle 6.309), agreement 96.4%; 18.5K 6.267 (oracle 6.267), 96.2%. The kernel's code size decided its speed: the first fused version (44 KB of code, above the 32 KB instruction cache) ran slower than the old kernel; IEEE divisions in the FP8 rounding replaced by exact exponent shifts and a rolled epilogue (36 → 21 KB) took the ratio-1 layer 17.6 → 13.0 ms, homogeneous tiles with table decoding (14 KB) → 9.4 ms; row ids read from global memory per row (26 KB) lost. No gain: the partial scores sharing the KV row buffer (three workgroups per WGP, one more barrier). |
| The FP16 routed kernel's weight-block loop rolled into one copy of the block body (the two-block ring shifts by register moves instead of two unrolled steps plus a tail copy): 40 → 16 KB of code per variant, below the 32 KB instruction cache | Retained. `ds41_mmq_bench --f16 1`, gate/up+SwiGLU and down per layer: IQ1_S 22.8 → 22.2 and 12.2 → 11.4 ms, IQ1_M 24.2 → 22.8 and 12.7 → 11.8 ms, IQ2_XXS 27.2 → 26.1 and 14.1 → 13.1 ms; routed kernels in the 2K trace 1436 → 1377 ms. Same values (`ds41.mmq_typed`); batched 3K perplexity 6.294, agreement 96.4%. |
| FP16 activations into and out of the dense FP16 kernel (the producer stores FP16, the kernel stages them without conversion), which would also halve those activation buffers | Not adopted. `ds41_mmq_bench --dense` at 2048 tokens, FP32 → FP16 input: q_b 5.23 → 5.66 ms, output LoRA 6.36 → 6.87 ms, o_b 5.43 → 5.88 ms, shared down 1.38 → 1.56 ms; only the shared gate/up gained (3.27 → 3.11 ms). The kernel is not bound by its activation reads. |
| Larger prefill batches for the routed experts (`ds41_mmq_bench --f16 1 --tokens 4096` on the recorded counts scaled ×2) | Available as `--prefill-batch 4096` (`serve`, `bench`, `prompt`); the default stays 2048. The skewed expert counts leave fewer padded 16-slot tiles at 4096 tokens: routed throughput 26 → 31.5 TOPS. End to end, with the FP16 matrix-core kernels: `gufo bench -p 4096` 694 → 720 tok/s (+3.7%, alternating runs), the first 4096-token prompt in a fresh session 618 → 682 tok/s, 18.5K prefill 31.5 → 29.7 s with perplexity 6.270 (oracle 6.267), agreement 96.2%; activations +1.12-1.21 GiB. 8192-token batches were slower (18.5K prefill 37.2 s, +3.4 GiB), so the option stops at 4096. |
| The dense FP16 kernel at 256 weight rows per workgroup (16 waves sharing one staged activation tile, which halves the activation reads) and its activations prefetched two stages ahead | Retained. Removing the activation loads (timing only) had taken the output LoRA 6.38 → 3.76 ms, while removing the weight decode or the fragment exchange saved 3-6%. `ds41_mmq_bench --dense` at 2048 tokens: q_b 5.23 → 5.21 ms, output LoRA 6.36 → 5.55 ms, o_b 5.43 → 5.14 ms, shared gate/up 3.27 → 3.11 ms, shared down 1.38 → 1.41 ms. 2K batch 3443-3608 → 3427 ms, 952-token batch 1845 → 1820 ms; batched 3K perplexity 6.294, agreement 96.4% (unchanged). No gain: 512 rows per workgroup (one workgroup per WGP), the stage loop rolled into one body (23 → 8 KB of code, 1-3% slower), a barrier that skips the wait for global loads. Without its two barriers per stage (timing only) the kernel ran the output LoRA in 4.20 ms and o_b in 4.83 ms, but double-buffered activation tiles with one barrier per stage were slower (o_b 5.14 → 6.34 ms with 64-value stages, one workgroup per WGP; 6.49 ms with 32-value stages, two per WGP). |
| Batched mHC state passes fused (`hc_state_pass_kernel`): the mixes and the collapse with its RMS norm read the state once (the collapse uses the pre-mix that was known before the mix, from the shifted mHC), and before the FFN the attention expand runs in the same pass, so the expanded state is not read back twice. The mix loop is rolled (34 → 7 KB of code) | Retained for 16 tokens and up. `ds41.hc_state_pass`: mixes and normalized rows within 3e-7 of the separate kernels, the state bit-identical. 2K batch 3427 → 3359 ms, 952-token batch 1820 → 1790 ms, 18.5K prefill 33.3 → 31.5 s (with the 256-row dense kernel); 18.5K perplexity 6.273 (oracle 6.267), agreement 96.2%; batched 3K 6.320, 96.0% (rounding drift through the FP8/FP4 fake quantization; earlier rounding-only changes spanned 6.285-6.326). |


## Post-M6 resident optimization pass (2026-10-26)

Baseline source: `ee49a0fd` (sampling defaults from `edce52dd`); production
`release` builds with the same Nix compiler/dependencies. The source review
is [OPTIMIZATION_REVIEW.md](OPTIMIZATION_REVIEW.md). Each experiment records
its disposition, including rejected attempts. Raw observations and provenance
for E1 are in [artifacts/2026-10-26-rescore](artifacts/2026-10-26-rescore/README.md).

| Experiment | Decision / evidence |
| --- | --- |
| E1: compact candidate blocks for FP32 indexer rescoring, 1–4 rows after 16K; keep dense score storage and the original final top-k | Retained for the measured AR gain: tg128 at 32K 22.78 → 23.05 tok/s (+1.2%), at 128K 21.10 → 21.89 (+3.7%); depth-zero control 23.97 → 24.03. Selected scores and all 512 output indices retain their original bytes/order; independent FP64 score error ≤2.2e-7 in the correctness cases. Uses the existing block-select scratch, with no extra persistent allocation. Full-history source indexing and WMMA prefill rescoring remain future work. |

E1 compacts layer 20's mask into increasing physical block IDs for each
reindex call and clears unselected dense scores to −inf. The existing FP32
score body gathers only those blocks. Keeping the original final selector
preserves both the selected set and its emitted order, including ties and a
partial newest block. A separate template specialization leaves unrestricted
source scoring on its original arithmetic path. The eight-decode 32K trace
confirms 32 compact calls, 32 unrestricted calls and zero old apply-mask calls;
GPU time falls 327.27 → 325.91 ms and span 362.24 → 360.86 ms. Each scorer
specialization uses 248 VGPRs, 128 SGPRs, 51,328 B LDS and no private scratch.

Qualification: `ds41.attention` and `ds41.candidate_mask` pass, including
varying FP4 scales, tied scores, different query masks, causal partial blocks
and exact-size allocations. Teacher-forced controls remain at perplexity
6.320 / 6.273 (3K / 18.5K). At 18.5K, an eight-cycle width-four speculative
probe retains 20 rows: every full logit vector is bit-identical both to serial
decode and after changing rejected suffixes. Details are in [QUALITY.md](QUALITY.md).

Long-context API correctness passes with and without the Q2_K/Q4_K drafter,
at 22,591 prompt tokens: sampled replay, prefix reuse, cancellation/resume
and output stop handling. AR passes all 124 per-request/phase timing checks.
DSpark performance is **inconclusive and unqualified**: three alternating
pairs and an unchanged-baseline control yield 86 passing and 38 inconclusive
metrics, zero confirmed slowdowns, and no quality/coverage changes. The
flagged cancellation queues and short replies accompany varying draft widths;
their observed timing ranges overlap. Every observation is retained at the
unchanged 5% / 3 ms margins. No DSpark speedup is claimed.

The C2 32K control (one repetition) measures 11.50 → 11.55 tok/s per user,
with aggregate prefill 1127.04 → 1122.41 tok/s. It is a concurrency control,
not evidence of a concurrency speedup.

The same C1 runs' prefill controls were 1187.67 → 1219.95 tok/s at depth zero,
1109.71 → 1123.29 at 32K and 856.10 → 829.19 at 128K. These paths are unchanged
by E1; their variance is retained, rather than counted as prefill improvements.

P0 profiling refresh: the final pp2048 at 32K, with decoder bounded replay,
uses 1,773.84 ms GPU time over a 1,805.14 ms span. Stage sums: routed FP16
606.39 ms, dense FP16 518.14, attention 219.06, mHC 137.38, indexer 82.95 and
top-k 45.34. Loading and depth preparation are excluded. The older backlog's
3.30-second prefill breakdown predates bounded replay and is not a current
end-to-end cost estimate.


### E2: stochastic DSpark proposals (retained, with confidence-gated verification)

Connected the existing bounded-q proposal and exact p/q residual callbacks to
post-Markov top-eight candidates, enabled unfiltered sampled requests, and
froze sampled cycle costs so wall-time variation could not change seeded
proposal draws. Each next Markov bias used the actual sampled proposal.

One focused check passed in 0.17 s: biased GPU top-eight with ties/non-finites,
independent target-frequency correction (including tokens outside q), and
seeded proposal/residual replay. A real 256-token candidate replay was
identical with tracing enabled; 20 probabilistic verification rows ran.

Matched IQ1_M + Lucebox Q2_K/Q4_K, temperature/top-p 1.0, seed 73, 256 tokens:
coding 36.335 → 37.527 tok/s (+3.3%), prose 23.469 → 23.045 (−1.8%). Coding
acceptance was 73.9 → 76.9%; prose 46.1 → 22.2%, with the candidate mostly
backing off to plain steps. Proposal draws change the continuation, so these
single-seed observations are not a general speed qualification. The prototype
is rejected for insufficient consistent gain; production retains point-mass
sampling. No broad functional matrix was run. The patch, settings, focused
check and every measured run are retained in
[the E2 record](artifacts/2026-10-26-sampled-proposals/summary.json).

**Re-measurement.** The single-seed trial was too small: the prose candidate
drafted 18 tokens in total, so its 22% acceptance is noise. Against
`8d51885a`, four prompts (coding, prose, explanation, Rust) × seeds
73/11/29, 256 sampled tokens and temperature/top-p 1.0, the original
prototype gains +1.8% geometric mean with its fixed costs and +2.5% with
adaptive costs. Pooled acceptance per drafted token rises from 0.656 to
0.675/0.686. Moving the draw to the GPU (below) adds nothing on its own
(+0.7% fixed, +2.4% adaptive). Per-pair SD is 5–10%, because each variant
samples a different continuation.

Forced five-token drafts (8 runs, 680 cycles) show where the cost goes.
Proposals with q's best probability ≥ 0.7 (point masses) are accepted 85% of
the time; diffuse ones only 19–51%, by confidence bin. A verified row costs
11–13 ms (spec-bench: 41.8 ms for one row, 99.1 ms for six), about 30% of
a plain step. That is routed-expert traffic: about 73% of the 1 → 4 row
increase, with `expert_group` near the measured read ceiling. Verifying
diffuse rows therefore costs more than it emits. An offline replay of the
traces with the measured costs ranks "draft the limit, verify up to the first
proposal with q's best < 0.6" above every fixed width (32.1 vs 27.9 tok/s
simulated over all cycles).

Retained design:

- Sampled requests (any temperature > 0, now including top-p 1.0) draft
  from the post-Markov top eight. The draw happens on the device: the host
  draws one 24-bit uniform per row up front from the request RNG, and one
  kernel per row finds the biased top eight, forms q (the temperature
  softmax in 2^-24 units, or a point mass when its best is ≥ 0.7) and writes
  the proposal where the next Markov row reads it. q returns once per cycle,
  and the bridge verifies against exactly that q with p/q acceptance and
  residual correction.
- Sampled cycles draft the request's limit and verify the prefix of
  proposals whose q best is ≥ 0.6; a cut at zero becomes a plain step.
  Widths depend only on q, never on timing, so seeded replay is
  deterministic. Repeated runs and the gated/fused builds give identical
  hashes. Greedy requests keep the cost-based controller.

Same 12-pair protocol: +3.85% geometric mean against `8d51885a` (9/12
faster; coding +4.7%, explanation +4.2%, prose +4.6%, Rust +1.9%). Pooled
29.76 → 30.93 tok/s; accepted per verification 1.80 → 2.07. The focused
check covers the biased top eight with ties/non-finites, device q (softmax,
mass sum, draw boundaries and the point-mass rule) and exact recovery of the
target distribution through Retain/Verify over 30,000 seeds, including
tokens outside q. Each configuration's runs, the traces, the replay
simulator and the variant patches are in
[the re-measurement](artifacts/2026-10-26-sampled-proposals/remeasure/summary.txt).


### E3: request-local BPE piece memoization (retained)

Cache previously tokenized pretokenizer pieces within each immutable text span,
following the existing Qwen/Gemma memoization approach. The cache holds at most
512 pieces of 2–64 bytes and activates for spans of at least 1,024 bytes. Borrowed
keys never outlive the input; cached IDs never cross a call or vocabulary reload.
Pretokenization, merges, Unicode handling and special-token handling are unchanged.

Every emitted token-ID line matches the baseline for eight copies each of a
repository documentation/source corpus with multilingual and combining-character
text, and a corpus of 12,000 unique identifiers. The metadata-only probe workload
(including process setup and ID printing) takes 172.53 → 149.19 ms for the first
corpus and 113.53 → 108.22 ms for the second. These are single-run host component
measurements; no model throughput or TTFT gain is claimed. Production and probe
builds and the C++ format check pass. No GPU model sweep was needed for unchanged
token IDs. Settings, corpus descriptions and output hashes are retained in
[the E3 record](artifacts/2026-10-26-tokenizer-memo/summary.json); raw inputs and
outputs are under `/tmp/gufo-ds41-tokenizer` on the measurement host.


### E4: stop unused DSpark Markov suffixes (retained)

Pass the controller's selected proposal count into the drafter. Keep the full
five-row transformer, normalization and output head, then stop the dependent
Markov gather/projection/argmax chain at that count and copy only those IDs.
The used proposal prefix retains its arithmetic; no target or rollback work changes.

The existing dump probe now checks widths 1–5 against the complete draft on a
128-row history, checks untouched caller output tails, and rejects widths zero
and six. It passes. Production and probe builds and the format gate pass.

C1 production CLI, 256 sampled tokens, temperature/top-p 1.0 and seed 73:
coding baseline 37.455 / clean control 37.873 tok/s, candidate 37.656; prose
23.219 / clean control 24.133, candidate 24.188. Each corpus has identical
output-token hashes in every run. Trace medians confirm short-width draft
latency falls: one proposal about 11.7 → 8.9 ms, two 11.6 → 9.7 ms, four
11.7 → 11.0 ms; five stays about 11.8 ms. Adaptive widths and plain-step
backoff vary between runs. **Aggregate throughput is approximately flat and
no general speedup is qualified.** Retained for removing measured unused work
with unchanged proposal prefixes; no larger quality sweep was run. All runs,
settings, trace distributions and focused checks are in
[the E4 record](artifacts/2026-10-26-markov-tail/summary.json).


### E5: score only requested draft-head rows (retained)

After E4, shortened the shared target vocabulary head to the requested draft
count, keeping the transformer and collapse normalization at five rows.
An extended dump check found every used vocabulary-logit row bit-identical
at widths 1–5 on a 128-row target history; proposal prefixes, untouched output
tails and invalid-width handling also passed. Production/probe builds and
formatting passed.

Matched to E4's candidate runs (same binary retained as this baseline), C1,
256 sampled tokens, temperature/top-p 1.0 and seed 73: coding 37.656 → 38.214
tok/s (+1.5%), prose 24.188 → 23.851 (−1.4%). Output-token hashes match for
both prompts. Short-width drafting is cheaper, but adaptive-controller
changes prevent a consistent total-request gain in this single-seed trial.
That single pair was first rejected; the trial was too small for a change
with identical outputs, since adaptive draft widths alone vary a run by
several percent.

Re-measured against `8d51885a` with the same production toolchain: four
prompts (coding, prose, explanation, Rust) × seeds 73/11/29, 256 sampled
tokens, temperature/top-p 1.0, alternating order without tracing. All 12
pairs emit identical tokens. Candidate throughput is +0.33% geometric mean
(9/12 pairs faster, per-pair SD 2.4%); pooled 29.78 → 29.87 tok/s. The
draft-prefix/logit probe passes again on the rebased change. Retained as a
small measured saving with unchanged output: the head now evaluates only the
requested rows, and the transformer body keeps all five. The paired runs
and scripts are in [the re-measurement](artifacts/2026-10-26-draft-head/remeasure-summary.txt);
the first trial remains in [the E5 record](artifacts/2026-10-26-draft-head/summary.json).


### E6: explicit FP32 index scoring for verification (retained for numerics)

Use the existing speculative-evaluation purpose flag to group index scoring
into at most four rows. This keeps the serial FP32 score/reduction at widths
five through seven and enables E1's compact rescoring there; ordinary prefill
keeps its WMMA dispatch. No kernel arithmetic or persistent-state format changes.
The probe now rejects non-finite logits and serial-logit mismatches, rather
than merely printing the serial comparison while gating only rejected-tail
independence. Correct the session comment that treated all sub-eight-row
kernels as inherently scalar-equivalent.

At 18.5K, baseline width six already matches serial on all 21 retained rows:
the dispatch gap did not demonstrate a model bug on this history. Candidate
width six also matches all 21 rows, and width seven at 3K matches all 28.
Each test exercises every commit length and corrupts the rejected suffix;
maximum logit differences from serial and between clean/corrupt passes are zero.
E1's independent FP64 scoring/selection evidence applies to the unchanged
FP32 kernel. Production/probe builds and the format gate pass.

One matched production C1 IQ1_M + Q2_K/Q4_K, greedy pp2048/tg128 pair:

| Phase | Baseline tok/s | Candidate tok/s |
| --- | ---: | ---: |
| pp2048 | 1108.95 | 1071.44 |
| tg128 | 23.69 | 26.12 |
| pp2048 at 32K | 1117.50 | 1118.19 |
| tg128 at 32K | 54.54 | 53.97 |

Both depth-zero and 32K generated-token hashes match. The 32K trace exercises
six-copy lookup plus anchor (seven verification rows), so the changed path
actually runs. Short decoding has adaptive-policy variation; prefill's code
is unchanged. No phase exceeds both slowdown gates (5% and 3 ms), but this
single pair does **not** qualify a speedup: the affected long-context decode
is about 1% slower. Retained as an explicit numerical contract for verification
and stronger qualification, not as a throughput win. Eight-row attention and
wider projection equivalence remain open; no verification limit was widened.
All checks, rates, hashes, commands and raw benchmark traces are in
[the E6 record](artifacts/2026-10-26-verification-scores/summary.json).


### E7: stable device route plans for FP16 layers (retained groundwork)

Large batches whose gate/up and down both use FP16 now build their expert
counts, bounds, stable slot lists and 64-slot tiles on the GPU. Reuse the
existing plan allocation and launch a bounded tile grid: unused entries are
−1 and return before reading expert metadata or weights. No added model or
scratch allocation. The IQ1_M artifact takes this path in 38 of 40 layers;
the two mixed-MMQ layers retain their CPU counts/classes and plans. Small
batches retain their existing GEMV plans.

The focused planner/operator check passes: exact rows, bounds and tile order
against the original host plan (with a standard-library stable-sort check),
63/64/65-slot tails, 1/7/384/512 experts, skewed routes, invalid and all-invalid
IDs, plus fused gate/up and down for IQ1_S/IQ1_M/IQ2_XXS. The 3K matched model
check retains all 2,999 argmax/NLL records at the probe's six-decimal NLL
precision, perplexity 6.319756. Generated-token hashes match. Production and
probe builds and formatting pass.

One production C1 pair (no drafter):

| Phase | Baseline tok/s | Candidate tok/s |
| --- | ---: | ---: |
| pp16 | 71.31 | 72.41 |
| pp128 | 362.46 | 362.58 |
| pp2048 | 1205.51 | 1205.80 |
| tg128 | 23.99 | 24.01 |

No phase exceeds both slowdown gates (5% and 3 ms), but these single pairs
qualify no throughput gain. A separate pp2048 trace confirms 38 device plans;
the three planner kernels total 1.18 ms. Excluding load/setup, GPU work is
1624.49 → 1621.79 ms, GPU span 1651.56 → 1645.07 ms; removing host plan/class
uploads offsets the new planning kernels. Retained for eliminating the 38
host round trips and making those layers' route planning device-resident,
with effectively flat unprofiled throughput. Graph capture and the mixed-MMQ
plan remain open. All observations and checks are in
[the E7 record](artifacts/2026-10-26-device-route-plan/summary.json).


### E8: 32-slot routed FP16 tiles (rejected)

Template the routed FP16 kernel on its tile width and launch a 32-slot grid
for device-planned tiles with at most 32 live slots, beside the existing
64-slot grid. Both consume the same 64-slot plan, so each live tile has one
owner. The 32-slot body needs 135–146 VGPRs and 15–25 KB LDS, against 169–180
VGPRs and 24–34 KB for 64 slots.

The first prototype measured pp2048 1210 → 562 tok/s. The cause was code
generation, not the tile width: its ownership early return lets the compiler
prove at least three live 16-slot groups in the 64-slot body, which then
needs 256 VGPRs, 538–671 spills and about 2 KB of private scratch per lane.
Hiding the count's range after the check restores the original 64-slot
resources exactly, with no spills in any variant.

Spill-free variant, two alternating production `bench` runs (no drafter),
against `8d51885a`: pp512 768.3/769.0 → 758.8/772.5 tok/s, pp2048
1243.8/1247.4 → 1232.9/1244.3, tg16 unchanged. Narrow tiles therefore give
no measurable prefill gain. Rejected; the 64-slot kernel is unchanged. Both
patches, per-kernel resource usage and the runs are in
[the E8 record](artifacts/2026-10-26-routed-32-slot/resource-usage.txt).


### E9: fuse the Markov projection into its selection (retained)

After E2, a five-row draft spent 4.25 ms in its Markov tail, more than the
2.48 ms vocabulary head. Each row ran the rank-256 FP16 W2 projection
through the generic matmul (686 µs for 66 MB, about 96 GB/s), wrote a
129K-entry bias, then scanned logits + bias in a single workgroup
(278 µs for the top eight).

One pass now computes each W2 row (one 16-byte load per lane), adds the
head logit and keeps the best one or eight per wave. Each of 256 blocks owns
a contiguous token range, and emits its candidates by descending value then
lowest token. A one-block merge (argmax, or top eight plus the E2 device q
draw) therefore keeps the lowest-token tie rule. Other W2 formats keep the
previous path.

The focused check compares the fused argmax/top eight with an FP64
reference over the full 129,280-token vocabulary. It covers NaN/−inf rows and
an exact tie split across blocks, which must resolve to the lower token.
Greedy outputs and acceptance statistics are identical to `8d51885a` for
coding and prose (the drafted tokens match); median sampled draft time falls
12.6 → 10.0 ms with an identical token hash.

12-pair sampled protocol (as E2): +2.09% over E2 (11/12 faster, per-pair SD
1.2%; the continuations are identical), +5.26% over `8d51885a` (11/12;
coding +4.2%, explanation +3.7%, prose +8.2%, Rust +5.0%). Pooled
30.10 → 31.73 tok/s. Runs and checks are in
[the E9 record](artifacts/2026-10-26-fused-markov/summary.txt).


### E10: FP32-activation GEMV for the drafter's Q4_K projections (retained)

The drafter's dense Q4_K projections (q_a/q_b, KV, output a/b and the shared
expert, about 87 MiB per stage) ran through Q8_1 MMVQ for five rows. They
now take the target's K-quant GEMV with a Q4_K format: FP32 activations,
each weight decoded once for every row. The target model's own dispatch is
unchanged.

The operator test adds Q4_K against the reference dequantizer: relative error
≤ 5.2e-7 for 1,280/5,120-wide rows at every slice count. Two to eight
tokens, pair and fused SwiGLU are bit-identical to single-token passes. Median
sampled draft time falls 10.0 → 9.3 ms. Draft arithmetic changes, so sampled
continuations differ; greedy outputs remain identical.

12-pair sampled protocol against E9: +0.41% geometric mean (7/12 faster,
per-pair SD 7.4%), pooled 31.89 → 32.14 tok/s. Acceptance per draft
0.682 → 0.693, positional 0.804 → 0.800; no acceptance loss is visible. The
end-to-end gain is within noise. Retained for the measured draft-stage
saving. Runs and checks are in
[the E10 record](artifacts/2026-10-26-draft-q4k-gemv/summary.txt).


### E11: learned DSpark acceptance head for greedy cycles (retained for greedy)

The drafter GGUF ships a learned acceptance head
(`dflash.dspark.confidence.weight`, 5,376 F16, zero bias): the sigmoid of
one dot product over the pre-norm head collapse of each draft row and that
row's Markov W1 input. It is now bound when present and evaluated per row by a
one-block kernel after the Markov gather. With the head, greedy cycles draft
the request's limit and verify the prefix whose rows each predict acceptance
≥ 0.5, replacing the timing policy's one width per cycle. Sampled cycles keep
the q-top ≥ 0.6 cut and do not run the head.

Forced width-5 traces (two seeds sampled plus greedy, four prompts) predict
acceptance far better than q-top: AUC 0.856 vs 0.735 sampled, 0.877 greedy
(q-top carries no greedy information), and the head is close to calibrated
(the 0.5 bin is accepted 64%, the 0.9 bin 99%). Simulated with measured verify
costs, the cut-at-first-row-below-τ policy is flat for τ 0.45–0.55 and matches
cumulative-survival and expected-rate rules within 1%.

The operator test compares the kernel with an FP64 reference, with and
without bias. Greedy outputs are identical to E10. Greedy protocol (four
prompts × three repetitions): +13.07% geometric mean, 12/12 faster
(coding +9.9%, explanation +22.9%, prose +9.3%, Rust +10.7%); pooled
29.87 → 33.94 tok/s, acceptance 0.757 → 0.798. The same cut on sampled cycles
measured −1.0% (3/11 faster, per-pair SD 4.9%): 13% fewer verified rows,
but accepted tokens per step fell 2.00 → 1.92. It is not used there, and
sampled outputs remain byte-identical to E10. Functional sampling, batch,
stop and tool suites pass with DSpark. Runs, traces and the simulator are in
[the E11 record](artifacts/2026-10-26-learned-confidence/summary.txt).

### E12: device target draws for sampled DSpark verification (retained)

A cycle split showed that copying verification logits to the host costs
0.07 ms, and greedy host verification about 1.3 ms per six-row cycle (not
worth moving). Sampled host verification cost about 5 ms per cycle: each row
built a 129,280-entry FP64 softmax (0.6 ms), and a rejection added a hashed
residual pass.

Sampled rows are now drawn on the device while the request's selection is
plain softmax sampling: positive temperature, no top-k, top-p 1, no min-p,
penalties, active grammar, forced reasoning end or deferred draw. Three
kernels compute per-span maxima, per-span FP64 softmax sums and the draw: p/q
acceptance of a proposal with two or more tokens, otherwise a plain draw, and
a residual draw after a rejection. The draw walks its span in token order
with the host sampler's arithmetic. The session draws the host's uniforms in
the host's order and restores the RNG to the consumed count. Rows the host
would draw without a uniform (fewer than two nonzero probabilities) and every
other configuration use the host sampler. The bonus draw after a full block
(or a plain step) uses the same path. The device argmax supplies the
positional statistic.

The operator test compares 240 device draws (97 accepted and 23 rejected
proposals, 120 plain draws, rows with NaN/−inf, temperatures 0.6 and 1) with
the shared host sampler: tokens, acceptance, uniform consumption and argmax
are identical, and a single-token row is reported degenerate. Sampled and
greedy outputs and speculative statistics are identical to E11. Full-width
verification falls 106.0 → 101.1 ms.

12-pair sampled protocol against E11: +1.99% geometric mean, 12/12 faster
(per-pair SD 0.6%; coding +2.6%, explanation +2.0%, prose +1.4%, Rust
+2.0%), pooled 32.58 → 33.21 tok/s, with identical outputs. Each device row
still costs three launches and a synchronizing copy, so the realized gain is
below the removed host time. Functional sampling, batch, stop and tool suites
pass with DSpark. Runs are in
[the E12 record](artifacts/2026-10-26-device-target-draws/summary.txt).

### E13: one launch set for all verification rows (rejected)

The E12 draw kernels gained a row dimension: every verification row and the
bonus row drawn in one launch set with one copy, the uniforms drawn up front
in the host's order with the RNG state after each, and rows handed out in
order with per-row eligibility rechecks. A five-row batch matches the
single-row draws exactly and outputs are identical to E12. 12-pair sampled
protocol against E12: −0.50% geometric mean (3/12 faster, per-pair SD 0.8%).
Drawing the rows after the first rejection costs about what the saved
launches and copies gain, so E12's per-row draws remain.

Plain DeepSeek V4.1 decode needs no separate device draw: with prompt lookup
always on, its plain steps run through the lookup/DSpark step and already take
E12's bonus draw (31 of 32 tokens in a traced run; the request's first token
is drawn on the host). A `SampleNext` path for the generic decode loops was
therefore dropped. Patches and runs are in
[the E13 record](artifacts/2026-10-26-device-draw-rows/summary.txt).

### E14: joint evaluation of concurrent DSpark sessions (retained)

Concurrent DSpark requests stepped one session after another, so every
session's verification read all weights again. A batch step now prepares
every session's block (anchor, prompt-lookup copy or draft), evaluates the
blocks of up to 16 rows as one joint speculative evaluation, then verifies,
commits and emits per session. Projections, routed and shared experts,
Engram, mixes and the output head run once over all rows; RoPE, the
compressor, the indexer and attention run per session against its own
state through activation views that start at its rows. Each drafter state
keeps its own copy of its device proposals, which the next session's draft
would otherwise overwrite. Batches with image tokens or another activation
set fall back to per-session evaluation.

Every row keeps single-token arithmetic. The dense FP32 GEMV and the
grouped expert GEMV take up to 16 rows (expert plans of 128 slots); a
decoder's `decode_rows` keeps ordinary decode and prefill on their 8-row
switch points and lets joint evaluations use 16. Two kernels changed
arithmetic at exactly 16 rows: BF16 projections switch to WMMA tiles, so
decode batches issue them in groups of eight; the per-token hc mix kernel
now covers 16 rows, which previously always took the state pass.

The kernel test matches 12- and 16-row launches against one launch per
row for every dense and expert type. `ds41_load_probe --joint-check`
evaluates 2–4 sessions of 1–6 rows each, jointly and each on its own: all
328 rows of the four-session run are bit-identical, and the joint
evaluation takes 201 ms against 292 ms for 13.7 rows. In `gufo bench`
(tg128 at depth 512, alternating builds), per-user decode rises 17.4 →
24.3 tok/s at C2 (+40%) and 8.6 → 13.8 at C4 (+60%) greedy, 13.6 → 19.1
and 6.8 → 11.0 sampled, with C1 unchanged and every request's output and
speculative statistics identical. Functional sampling, batch, stop and tool
suites pass with four sessions; their C4 timing rows stay inconclusive, as
an unchanged-build control flags the same arrival-order-dependent cases.
Runs are in [the E14 record](artifacts/2026-10-26-joint-sessions/summary.txt).

### E15: 16-row single-session verification (retained)

Prompt-lookup verification stopped at seven rows (the anchor and six
copies): from eight rows the matrix-core attention replaced the decode
kernel and changed logits. On a code-edit reply, every lookup cycle copied
and accepted six tokens, so the cap set the speed. Split attention, which
evaluates each row on its own, now runs up to 16 rows whenever the caller
supplies its partial buffer, and a speculative evaluation uses the decode
switch points up to 16 rows, as E14's joint evaluations do. A session now
verifies the anchor with up to 15 copies; DSpark blocks are unchanged.
Ordinary evaluations of exactly eight rows, which already used the per-row
GEMVs, now also take split attention.

The attention test adds 12- and 16-row split cases against the f32
reference. `ds41_load_probe --spec-check` keeps every row bit-identical to
single-token decode at widths 8, 12 and 16 on the 1,096-token prompt, and at
width 16 on the 3,000- and 18,500-token histories (136 rows each). A
verification costs 113.9 ms for 7 rows and 223.1 ms for 16, so fully
accepted lookup commits 71.7 instead of 61.5 tokens per second. Repeating a
3,000-character source file with one rename (greedy, 900 tokens) rises from
58.5 to 67.6 tok/s (+15.6%) with 62 instead of 131 verifications, every
copy accepted and identical output. The four standard prompts, which copy
nothing, are unchanged and identical in sampled and greedy runs, and so is
prefill. The `long-context` and `cache-edits` suites pass without quality
changes; one first-run decode flag did not recur on two reruns or two
unchanged-build controls, and the remaining prefill flags match the
controls' variance and stay inconclusive.
Runs are in [the E15 record](artifacts/2026-10-26-wide-verification/summary.txt).
