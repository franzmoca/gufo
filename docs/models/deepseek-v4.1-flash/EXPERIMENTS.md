# DeepSeek V4.1 Flash experiments

Everything measured on the DeepSeek V4.1 Flash engine, retained or rejected,
with its evidence. Open work is in [BACKLOG.md](BACKLOG.md); the streaming
design is in [STREAMING.md](STREAMING.md); quality checks are in
[QUALITY.md](QUALITY.md) and the benchmark card in
[BENCHMARKS.md](BENCHMARKS.md).

Target: DeepSeek-V4.1-Flash-1017 with the
[ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
on one Strix Halo (gfx1151, 124 GiB GTT). The engine is
`src/models/deepseek_v41_flash/`, a fork of the V4 Flash engine (unchanged);
`tools/ds41/` holds an independent CPU/torch oracle that ports the official
`inference/model.py` and `inference/engram.py` and reads the same GGUF.

## Milestones

| Milestone | Result |
| --- | --- |
| M0 Oracle and loader | `tools/ds41/reference.py` (official KV quantization: FP8 SWA, FP4 compressed and indexer; chunked sparse attention with sinks, candidate blocks and Engram; dumps logits, routes and per-layer states; `verify-engram` matches the GGUF's hashing tables exactly) and `tools/ds41/quants.py` (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S and IQ4_NL dequant, bit-exact with llama.cpp). Oracle on IQ1_M: 3K-token teacher-forced perplexity 6.31, top-1 58%. The loader maps the split GGUF as one range and validates all `deepseek41.*` metadata and 1,008 tensors; the tokenizer matches the official `tokenizer.json` on ~19k tokens. |
| M1 Correct AR decode on IQ1_M | Single-token decoder over plain kernels with typed MMVQ entries for every expert and dense type; sessions with prefix reuse and snapshots; the official V4.1 chat template pinned by SHA-256 (a GGUF with another template loads with a warning); `deepseek41` in `gufo serve` with DSML tools and the official sampling defaults (temperature 1.0, top-p 1.0). 3K perplexity 6.306 against the oracle's 6.309, 12/12 template goldens, decode 12–14 tok/s. |
| M2 Resident performance | `gufo prompt`/`bench` support, candidate-block parity beyond 16K, batched layer-major prefill on the FP16 matrix cores, overlapped Engram reads (0.04 ms wait per prefill batch, 0.33 ms per decode token): pp2048 703 and tg64 23.9 tok/s at depth 0, 372 and 21.0 at 131K. Steps in [the resident engine table](#resident-engine-m1m5). |
| M4 Component cache and serving | Compressed KV and index K as append rows (ratio-2 layers add a row every 2 tokens); SWA rings, compressor frontiers and logits as private state; the Engram n-gram history rebuilt from the tokens on restore. Disk cache and prefix reuse as for Gemma 4. |
| M5 DSpark, prompt lookup, bounded replay | `--dspark-model` loads the unsloth (`dflash`) or Lucebox (`deepseek41-dflash-draft`) layouts; drafts of up to 5 tokens verified with decode numerics, so output is unchanged: greedy +21% on a document prompt, +61–67% on a coding request. Prompt lookup copies up to six (later 15) tokens: file edit 23.6 → 54.6 tok/s without a drafter, 43.9 → 56.4 with DSpark. Decoder SWA bounded replay: pp2048 703 → 1,177 tok/s, 372 → 855 at 131K, continuation NLL within 0.008 of exact prefill (`--exact-prefill` keeps every row). Not used: unsloth BF16 (7.92 GiB), Lucebox `-MXFP4-Q8.gguf` (7.42 GiB), the peasantsmith `.bin` drafters (not GGUF). |
| M6 Vision | Opt-in `--mmproj` (0.90 GiB); the sidecar adds each MoE gate's image routing bias (`bias_vl`) from the official checkpoint (`tools/ds41/vision_sidecar.py`). Official preprocessing (padded pixels byte-identical to PIL), the 32-layer ViT and 3 × 3 aligner in the official BF16 rounding (`tools/ds41/vision_reference.py` is the CPU teacher); image positions take no Engram term and route with the image bias. [Quality](QUALITY.md#vision-m6), [speed](BENCHMARKS.md#vision-encoder). |
| M3 SSD expert streaming | `IQ2_XXS` streaming qualified for evaluation; [below](#ssd-expert-streaming-m3). |
| M7 Higher-precision Engram tables | Open ([backlog](BACKLOG.md)). |

The inherited V4 `runtime/session.cpp`, `runtime/rocm_graph.cpp`,
`dspark_internal.h` and `dspark_policy.h`, which the V4.1 target never
built, were removed on 2026-10-27; the active V4.1 runtime is
`v41_decode.cpp`, `v41_session.cpp` and `v41_dspark.cpp`.

## Measured limits

- Measured DRAM read peak (`tools/bench/gfx1151_peak.hip`): 235 GB/s. The
  dense decode projections run at 207-221 GB/s on the large shapes; decode
  spends 42.6 ms of GPU time per token at 2K (47.2 ms wall).
- Routed expert matvecs: a plain read of the same six DRAM-cold 2.3 MB
  experts streams at 230-238 GB/s (58 µs); the pre-October-27 expert GEMV
  reached 160-175 GB/s on IQ1 and 121-134 GB/s on IQ2, and mmvq 145-170
  GB/s, bound by the texture addresser (`TA_BUSY` ~92% of active cycles):
  every 32-value sub-block cost four 2-byte weight loads and eight
  activation loads per lane. The retuned kernels
  ([expert GEMV decode retune](#expert-gemv-decode-retune-2026-10-27))
  decode at 162-205 GB/s, the dense GEMV's level for these shapes.
- Streamed `IQ2_XXS` expert reads: 5.7 GiB/s effective, 8.2 GB/s for purely
  sequential reads on the 990 PRO ([storage reads](#storage-reads)).

## Resident engine (M1–M5)

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
| Multi-token dense Q5_K GEMVs (2-16 tokens, every slice count; not the fused pair or SwiGLU launches): one wave per Q5_K task over 32 rows, one row per lane, so a wave's activations are uniform and arrive through scalar loads; workgroup (tile, slice) runs the slice's blocks in order and reduces the tasks in the single-token kernel's xor-tree order through LDS, and the tile's last workgroup (atomic counter, 16 MiB scratch on the null stream) adds the slice results in wave order (`split_gemv_kernel`) | Retained. At 8 tokens the lane layout spent its time on activation vector loads, not weights: feeding all tokens from one token's activations (same FMAs, wrong results) took wo_b 222 → 138 µs against 131 µs for one token. Bit-identical to single-token rows (`ds41.mmq_typed`, now with 2 slices and a partial last tile; `tools/bench/ds41_dense_gemv_tune.hip`). DRAM-cold, 8 tokens: wo_b 221 → 146 µs (130 → 198 GB/s), wo_a 166 → 123, q_b 211 → 139, wq_a 40 → 30, shared down 70 → 49; one token keeps the lane layout. `--spec-bench` on resident IQ1_M, 1-8 rows, alternating pairs: 40.9/52.6/67.7/77.1/92.8/102.3/117.5/129.7 → 41.0/51.9/65.5/73.7/87.4/96.7/107.9/118.8 ms (8 rows −8%). `gufo bench` tg512 with DSpark, three alternating pairs: resident IQ1_M 29.17/29.42/29.38 → 29.71/29.98/29.95 tok/s (+1.9%); streamed IQ2_XXS 33.17/34.14/33.28 → 33.15/33.93/34.73, inside its cache-warming spread. Rejected on the way: two or four rows per lane sharing activation loads in the lane layout (wo_b 8 tokens 222 → 238 / 294 µs); task-major with every slice in one workgroup (wo_b 219 → 289 µs, wo_a 167 → 192 µs: one wave per task walks the whole row, too few waves to hide latency; a round of 8 block loads per lane helped one token, 211 → 169 µs, not eight). |
| The expert-group plan built in O(slots²) (each slot finds its group's first slot, leaders count earlier leaders) instead of O(slots³) | Retained. Same plan (`ds41.mmq_typed`); 40 plan launches per 6-row pass 1.18 → 0.11 ms; `--spec-bench` 6 / 7 / 8 rows 97.1 → 96.1, 109.6 → 107.7, 110.8 → 109.3 ms. |
| Prompt lookup in DSpark sessions: a cycle whose context repeats earlier tokens (PromptLookup gates: 12 matching tokens, 24 and prompt-only for sampled requests, copies only as far as competing occurrences agree) verifies up to six copied tokens instead of drafting | Retained. File edit, 1,302 tokens, 512 generated: 43.88 → 56.43 tok/s, 71 of 78 cycles copied, 414 of 420 copies accepted; roadmap 28.95 and chat 37.81 tok/s (0 and 3 lookup cycles). Seven copies (eight rows) changed the output: the batched attention kernel starts at eight rows. |
| Decoder SWA bounded replay: prompts run layers 20-39 over their last 140 positions (DeepInfer's V4.1 default), layer 20 still compresses every position, and the replayed windows start at the first replayed position | Retained. Prefill 1.77-1.97× faster (8,863 tokens 14.70 → 7.75 s; 23,848 tokens 40.37 → 22.84 s); continuation NLL within 0.008 of exact prefill and top-1 agreement 254-255 of 256 ([QUALITY.md](QUALITY.md)). Serve: 11,327-token prompt 16.95 → 9.52 s. |
| Prompt lookup without a drafter, through the same speculative step | Retained. File edit 23.59 → 54.61 tok/s, chat 23.81 → 24.30 tok/s, greedy output identical. The first version evaluated each emitted token before returning it: serve's `cache` suite showed decode 37-39 → 42 ms per token and the second token 43 → 86 ms after the first, one evaluation more per request than plain decoding (which emits a token, then evaluates it). Steps now emit the token after the accepted copies unevaluated and verify it as the next step's first row: against the previous build, all 479 per-request timings pass or stay inconclusive (single flags that did not repeat over three alternating pairs), and the 30 completed replies are identical. With the DSpark drafter, five alternating pairs left only flags that changed direction between pairs, except one disk restore after a restart that waited 22-88 ms for a transfer stream in three candidate runs and 39 ms in one unchanged-build run (about 2 ms otherwise): inconclusive, and present without this change. |
| Grouped routed-expert kernel for 2-8 verification rows (IQ1_S gate/up is 17.9 ms of a 6-row pass for 21.2 distinct experts per layer; its weights alone would take about 13.4 ms at the single-row kernel's 203 GB/s) | Not adopted, all slower: the next weight block loaded across loop iterations (routed total 46.8 → 48.2 ms); per-group activation sums precomputed once per column (no gain); 10 waves per SIMD (177 → 144 VGPRs) 17.9 → 20.5 ms; two rows per lane sharing each activation load 17.9 → 45.9 ms (register pressure); 2-column tiles in a runtime loop (24 → 10 KB of code, 177 → 161 VGPRs) 17.9 → 19.5 ms. Counters: the 6-row kernel issues 4.3× the VALU instructions of the single-row kernel for 3.5× the experts, at about 55% VALU issue. |
| Vision encoder linears: hipBLAS `GemmEx` (BF16, small-tile kernels), Flash-Next's dense BF16 WMMA GEMM, hipBLASLt | hipBLASLt retained. Carrots encode: hipBLAS 1,460 ms, WMMA GEMM 1,030 ms; with the fused attention, WMMA GEMM 302 ms and hipBLASLt 249 ms. 4000 × 3000 with the fused attention: WMMA GEMM 1,252 ms, hipBLASLt 853 ms. Linears at 29 TFLOPS on the capped image. Spans within BF16 noise of the official forward. |
| Vision attention fused (online softmax, BF16 WMMA, 16 queries per wave) instead of tiled score GEMMs, softmax and value GEMMs | Retained. Carrots 1,030 → 302 ms. |
| Fused vision attention computing S^T = K Q^T and O^T = V^T P^T, so each lane keeps one query's softmax state and the half-waves swap probabilities (`permlanex16`) instead of shuffles and an LDS round trip; branchless BF16 packing; K/V tiles staged in LDS for eight waves | Retained. 4000 × 3000: 853 → 742 ms; carrots 249 → 236 ms. Bit-identical between the transposed variants. Variants: the hardware exponential (same output, 2%, within noise); LDS staging before the transposition (slower, the kernel was ALU-bound with 256 VGPRs); 16 waves (783 ms); register prefetch of the next tile (763 ms). |
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
`release` builds with the same Nix compiler/dependencies. The pass followed a
source review against DeepInfer (`5537415`, ROCm PR #201 `45ccedf`),
deepseek-recipe, ds4 and its Strix Halo PR, Kyojin, the Spark EXL3 recipe,
Project Maya, Xolotl and Gufo's Qwen/Gemma paths; its proposals that remain
open are in the backlog. Each experiment records its disposition, including
rejected attempts. Raw observations and provenance
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

The head is retained; its fixed ≥ 0.5 cut was replaced by the cost-aware
prefix of [E18](#e18-cost-aware-draft-prefix-retained).

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

### E16: 4096-token prefill batches after bounded replay (rejected as default)

Bounded replay changed what a prefill batch costs, so the 4096-token batch
was measured again. Natively it gains 2.0–2.2% at depth 0 and 2.8–3.1% at
32K for prompts of 2,560 to 8,192 tokens (before bounded replay: +3.7%).
Served in the default resident setup (native 1M context, one session,
Q2_K/Q4_K drafter), its extra activation memory leaves the prompt cache no
backing: every capture is refused, and a 2K continued prefill gains nothing
(970.3 against 968.5 tok/s). 2048 stays the default; 4096 remains an option where
memory allows a cache beside it (not measured).
Runs are in [the E16 record](artifacts/2026-10-26-prefill-batch/summary.txt).

## SSD expert streaming (M3)

All on the 128.75 GiB `IQ2_XXS` build with a Samsung 990 PRO unless stated;
`gufo bench` pp2048/tg128 with synthetic tokens, alternating binaries and
targeted page-cache eviction of the model shards between runs. The design
as it stands is in [STREAMING.md](STREAMING.md).

### E17: streamed IQ2_XXS reader and mixed-quant kernels

The first streaming path serialized expert reads and staged all selected
weights. Four parallel readers improved a matched cold pp2048/tg128 from
144.5–144.8 / 5.77–6.01 to 172.6–172.9 / 6.80–7.16 tok/s. Aligned direct
reads improved the four-reader result to 213.8–214.1 / 8.97–8.98 tok/s;
unaligned end-of-file reads and the buffered fallback are covered by the
expert-store test. A repeated warm prefill/decode gives the same rates with
buffered and direct I/O. Four readers beat one, but eight and twelve did not
improve the per-layer direct-read test. `io_uring` and polling offered no
measured benefit over threads in a sustained raw-read control.

The GGUF mixes IQ2_XS with IQ1_M and IQ2_XXS. Its new IQ2_XS FP16 routed
matrix-core kernel halves the measured 140- and 2048-row expert projection
time; on cold pp2048, it gives 217.5–218.1 against 213.9–214.5 tok/s and
preserves the independent 3K scoring quality. Starting selected-expert reads
while the independent shared expert computes raises that result to 219.6–219.8
tok/s and decode from 8.84 to 9.02 tok/s. Both overlap controls have identical
generated-token hashes; a 16-row full-logit and route comparison is byte-exact.
The prior generic and FP16 prompt paths have different rounding, so their
score files are compared to the oracle rather than claimed byte-identical.
[Storage reads](#storage-reads) ·
[quality](QUALITY.md#streamed-iq2_xxs-m3-preliminary).

### Storage reads

October 26, 2026: Samsung 990 PRO, XFS mounted `noatime`, kernel 7.2.11,
NVMe scheduler `none`, 4096 KiB read-ahead; no global kernel or filesystem
setting changed. `tools/ds41/stream_io_bench.py` reads actual IQ2_XXS expert
offsets in 40 layer-sized batches of 1, 2 or 6 missing experts, 1.02 GB of
payload per arm after dropping the shards' clean pages, two rounds in
opposite orders:

| Read mode | Readers | Payload GiB/s, two runs | Median layer ms |
| --- | ---: | ---: | ---: |
| buffered | 1 | 3.04, 3.11 | 5.22, 5.14 |
| buffered | 4 | 5.33, 5.31 | 3.05, 3.08 |
| buffered, `POSIX_FADV_RANDOM` | 4 | 5.27, 5.41 | 2.97, 3.02 |
| direct | 4 | 5.68, 5.89 | 2.65, 2.66 |
| direct | 8 | 5.81, 5.75 | 2.74, 2.73 |
| direct | 12 | 5.63, 5.68 | 2.69, 2.73 |

Four readers reach the useful queue depth; random advice was inconclusive.
The end-to-end effect of direct reads is in E17 above; on a repeated
identical prompt the second prefill (938 tok/s) and decode (13.44 tok/s)
were the same with either mode. Kyojin's
[`uring_bench.c`](https://github.com/Yamz-Labs/kyojin/blob/0c5b91a0686acc8129190f753e032715f0b0dcc2/tools/tier/uring_bench.c)
(3 MiB random direct reads, queue depth four, 16 GiB of one shard, two
seconds) reached about 4.9 GB/s with `pread` threads, `io_uring` and polling
`io_uring` alike (p99 about 3 ms), so `io_uring` was not adopted. The
global scheduler and read-ahead were not tested (not writable by this user).
A `rocprofv3` pp140/tg16 trace with buffered reads showed kernels occupying
1.36 s of the 9.76 s from the first router to the last kernel, the largest
gaps around expert staging: kernel throughput alone does not predict
streamed speed.

**Read-path profile (2026-10-27,** `522c8653`**).** `/proc/<pid>/io` sampling
around the bench phases plus `tools/ds41/engram_row_bench.py`:

| Read path | Pattern | Mechanism | Measured |
| --- | --- | --- | --- |
| Expert weights (demand + prefetch jobs) | per layer, three ~1 GiB blocks (gate, down, up), expert-strided within each | O_DIRECT `pread`, 4 workers, ≤32 MiB chunks | 5.7 GiB/s effective; 8.2 GB/s pure-sequential ceiling |
| Engram rows | 144 B at hashed-random offsets, 2 layers × 24 rows per token | buffered `pread`, ≤32 threads, page cache | 278 K rows/s cold (~1.06 GiB/s at 4 KiB granularity) |

A pp2048 batch moves ~4 GiB of expert reads as ~140 large IOs and 98,304
Engram rows as 4 KiB IOs: the Engram is under 10% of the bytes but ~99% of
the operations. `rchar == read_bytes` for whole runs: once the expert cache
has grown (its chunk growth reclaims exactly the clean file pages), the
Engram's page cache scores no hits, so every prefill batch spends ~0.35 s of
device time on random Engram reads that take queue depth from the expert
reads. This fits the fuller-cache-slower-prefill pattern of the compact
stage arms below, the 98 GiB-cache build's faster prefill and the DSpark
runs (4.7 GiB more free memory).

Two disks, via symlinks to the second NVMe (KIOXIA EXCERIA 1.8T, 2.3 GB/s
sequential against the 990 PRO's 8.2):

- **Engram and shards 3–4 on the KIOXIA: pp2048 113–449 tok/s.** Shards are
  layer-major, so those layers read at the slow disk's pace and gate every
  phase that touches them.
- **Engram alone on the KIOXIA: pp2048 724.9–727.3** (739.9–747.9 on one
  disk), decode unchanged: its random-read latency costs more than the
  contention it removes.

[I/O bench](artifacts/m3-streaming/io-bench.json) ·
[profile](artifacts/m3-streaming/profile-iq2-stream.log) ·
[io_uring control](artifacts/m3-streaming/io-uring-bench.log) ·
[direct-I/O candidate](artifacts/m3-streaming/bench-iq2-direct-io-candidate1.log) ·
[buffered control](artifacts/m3-streaming/bench-iq2-buffered-io-control1.log)

### Overlapped staging (2026-10-27)

Against the October 26 tree (four direct readers, stage copies of resident
experts, shared-expert overlap), same machine, model, histogram sidecar and
`gpu-test` binary family:

| Change | Decision / evidence |
| --- | --- |
| Read resident experts through per-expert pointer tables instead of staging copies | Retained: removes up to ~57 MB/layer of device-to-device copies per decode token. `ds41_gemv_bench` puts the table indirection at parity with the packed layout, bit-identical; `ds41.expert_store` checks table contents, eviction and admission on synthetic shards. |
| Split gate+up launches so resident experts compute while misses stream | Retained: the overlapped launch zeroes the read rows and the post-read launch completes them; each row keeps its single-launch arithmetic (Kyojin skips this lever because its separate launches change the FP32 sum order), so tg128 hashes `04842be7…` on both builds, AR and DSpark (110 drafted / 71 accepted in both). |
| Per-reader upload streams instead of blocking legacy-stream copies | Retained: uploads overlap the layers' kernels. |
| Runtime table-pointer choice inside the expert GEMV kernels | Rejected: rescheduled the IQ2 loops with 249 inserted `s_delay_alu` (914 → 1,141 instructions for IQ2_XXS), −4.2% resident tg128 (24.06 → 23.06 over six alternating pairs). Specializing the indirection at compile time (`kTable`) restores the resident address arithmetic: every `ds41_gemv_bench` case matches (81.1/85.7/88.0/93.7/151.0/146.3/156.2 µs) and resident pairs return to 23.9–24.1 tok/s. |

Matched `gufo bench` (seed 19 s, 82.8% route hits):

| Build | Prefill tok/s | Decode tok/s | DSpark decode tok/s |
| --- | ---: | ---: | ---: |
| October 26 (stage copies, shared overlap) | 592.1 | 11.65 | 13.50 |
| Tables, split overlap, upload streams | 664.7–669.6 | 14.99–15.08 | 17.92–17.96 |

Decode host wait 6.51 → 5.25 ms/token at unchanged route hits, so the gain
is overlap, not cache policy. Resident `IQ1_M` pp2048/tg64 (1070/23.0)
stayed inside the previous build's spread (945–1129 / 22.5–23.7). IQ2_XS
kernel checks: routed FP16 22.7 TOPS at pp2048 shapes (98% of IQ2_XXS),
decode GEMV 112–116 GB/s with the table mode bit-identical. Qualification:
`ds41.expert_store`, `ds41.mmq_typed`, the format check, the native-context
probe (106.4 GiB RSS peak, 8.7 GiB available at 1M context) and the tools,
conversation, sampling-defaults and batch HTTP suites with the Lucebox
drafter. [Logs](artifacts/m3-streaming/bench-iq2-tables-ar1.log)

### Expert GEMV decode retune (2026-10-27)

The IQ2 expert decode kernels ran at 121-134 GB/s, half the dense GEMV's
207-221. Two limiters, found with `tools/prof/isa_mix.py`: the IQ2 decode
cast each `uint64_t` grid entry's extracted byte straight to float, which
lowered to a quarter-rate `v_ldexp_f32` (129 per loop) plus 64-bit shifts,
and the sign ternary emitted a compare/select pair per element; the loads
were also fragmented (five 2-byte weight loads and 16 activation loads per
32-value task in the SwiGLU pass). External references agree on the failure
mode: the sanssushi and MBufo Gufo forks measure 102-118 GB/s on their
per-lane grid-LUT IQ2 decode against a 214 GB/s Q2_0 kernel at the same
216 GB/s peak.

| Experiment | Decision / evidence |
| --- | --- |
| Retuned all five formats in `ds41_expert_gemv.hip.cpp`: task payloads as whole-word loads (memcpy from the 2-byte-aligned blocks), IQ2 grids and signs staged in LDS as u32 word pairs, the IQ2 decode on the entry's u32 halves with signs applied by flipping the float sign bit, and the fused gate/up pass loading x once per block into registers (AccR); U=2 for IQ1_S and the IQ2 formats | Retained, bit-exact by construction and by measurement. Standalone `tools/bench/ds41_expert_gemv_tune.hip` (production TU included, rotating DRAM-cold experts): IQ2_XXS gate/up 143.0 -> 194.9 GB/s, down 123.9 -> 189.4; IQ2_XS 148.9 -> 205.1 / 133.5 -> 189.9; IQ2_S down 143.5 -> 200.7; IQ1_M gate/up 171.5 -> 194.5; IQ1_S down 153.3 -> 162.1 (its gate/up already streamed at 200.3). `ds41_gemv_bench`: fp32 162-201 GB/s, table mode bit-identical; `ds41.mmq_typed` and `ds41.expert_store` pass; 0 VGPR spills (grouped kCols=16 peaks at 218 VGPRs). |
| Fast-math contraction context in the IQ1_M tail | A pragma-free re-expression of `acc + fmaf(s1, d01, s2*(d23))` reproduces the production bits; freezing it with `#pragma clang fp reassociate(off)` does not (production's own emission reassociates). The port keeps the source expression verbatim; the tune bench's replica variant guards it. |
| Alternating matched `gufo bench` pairs (gpu-test binaries, seed 19, two controls and two candidates interleaved) | Streamed IQ2_XXS AR tg128 15.90/16.06 -> 16.57/16.94 tok/s and DSpark 18.54 -> 19.34 (71 accepted and token hash 04842be7… unchanged; 110 -> 107 drafted under the timing-adaptive drafter); decode host wait 5.4-6.5 -> 5.2-5.6 ms/token; resident IQ1_M tg128 23.55/23.36 -> 23.86/24.38 (hash 70d5a48a… unchanged). Prefill untouched (routed_f16 path). |

### Post-retune kernel survey (2026-10-27)

After the expert GEMV retune, a rocprof pass over a streamed DSpark
pp2048/tg64 run (4.15 s GPU busy) ranked every remaining kernel family and
the survey measured each candidate's headroom. The model tensor mixes were
read with llama.cpp's gguf-py: the streamed build's routed experts are
gate/up IQ1_M x16, IQ2_XXS x17, IQ2_XS x7 and down IQ1_M x28, IQ2_XXS x8,
IQ2_XS x4; the resident build gate/up IQ1_S x28, IQ2_XXS x9, IQ1_M x3 and
down IQ1_S x27, IQ1_M x9, IQ2_XXS x2, IQ2_S x2; the Lucebox drafter's
experts are Q4_K x3 and Q2_K x3 of six layers.

| Candidate | Decision / evidence |
| --- | --- |
| Dense Q5_K FP16 prefill (`ds41_dense_f16.hip.cpp`, 493 ms / 11.9% of the run) | Kept. 32.2-33.5 TFLOP/s across q_b/o_b/sh_gate/wo_a (`tools/bench/ds41_dense_f16_tune.hip`), WMMA-issue bound (128 WMMAs against ~2.1k VALU per body, 615 s_delay_alu); the iterative-ILP scheduler flag measured 32.1-33.4 with identical hashes (rejected), and the tile/stage knobs are the ones the Gemma 4 audit already exhausted on the same kernel class at 37-40 TFLOP/s. The remaining ~15% is scheduling overlap, not instructions. |
| LDS-staged activations for the IQ1 down GEMV (162-167 GB/s, the weakest decode kernels; 28 of 40 streamed down layers are IQ1_M) | Rejected: the per-lane 128-byte activation stride makes every read an 8-way LDS bank conflict -- 43-49 GB/s, 4x slower than the global path. The down launches also show a fixed per-launch floor (85-96 us across 13.8-22.7 MB and all five formats), so activation traffic is not what holds them at ~165 GB/s. |
| DSpark drafter expert projections through the vendored mmvq (`mul_mat_vec_q_moe<10>` 49.8 ms + Q4_K paths ~12 ms + q8_1/fp4 passes ~19 ms of the run) | Not attempted: ~80 ms of a 4.15 s run. The drafter is not the DSpark bottleneck; the target's verification and head dominate. |
| Dense Q5_K FP16 prefill: double-buffered activation LDS (`ds41_dense_f16.hip.cpp`) -- stage g stores into buffer g&1 while stage g-1 still reads the other, one `lds_barrier` per K stage instead of two | Retained, bit-exact (output hashes equal at n=2048 and the ragged n=1904 tail, `tools/bench/ds41_dense_f16_variants.hip.cpp`). `ds41_mmq_bench --dense`: q_b 32.9 -> 33.6, o_b 33.5 -> 33.9, sh_gate 27.5 -> 28.1, wo_a 32.4 -> 32.5 TFLOP/s. A 64-token tile was 2.4x slower (matching the Gemma audit's tile sweep) and the iterative-ILP scheduler flag measured 32.1-33.4 with identical hashes (rejected). |
| DSpark drafter routed experts through the FP32 expert GEMV: new Q4_K and Q2_K traits in `ds41_expert_gemv.hip.cpp` (one 16-byte header + one 16-byte payload load per Q4_K task; Q2_K reads its 16 scale bytes at 0, the 32 bit-planed payload bytes of its plane group, and d/dmin at 80/82), and `moe()` fused into one gate/up/SwiGLU plus one down launch | Retained. Kernel level: Q4_K 216-217 GB/s, Q2_K 204-212 GB/s against the FP64 host reference (rel 3e-7, `ds41_expert_gemv_tune`), vs the mmvq path's ~100-150; two launches and the Q8_1 quantization passes drop. E2E (alternating matched pairs): streamed DSpark tg128 19.08/19.15 vs 19.08 (one control pair was cold-cache at 12.3), resident 26.26-26.43 vs 26.34-26.38, AR regression 16.74 vs 16.83 -- parity within noise, as the drafter is ~1.5% of the run. Outputs bit-identical (04842be7... / 70d5a48a...), acceptance identical (71/51); sampled sessions will draw slightly different drafts (FP32 vs Q8_1 activations), which verification keeps correct. |
| Q2_K trait first cut | Rejected and fixed: it assumed {d, dmin, scales, qs} with consecutive payload packing; the real block (`block_q2_K`) has the scales at 0, bit-planed qs at 16 (value v's bits in byte 16 + v%32 + 32*(v/128), plane 2*((v/32)&3)) and d/dmin at 80/82. The standalone FP64 check had passed because its reference and synthetic generator repeated the same wrong layout; kernel references are now checked against the vendored struct or gguf-py. |
| Persistent row-tiles / wider workgroups for the IQ1 down GEMV | Rejected / not ported: row-tiles (2 or 4 per workgroup) serialize work behind one table fill and lose 8-12% to reduced parallelism; 512-thread workgroups gain only the IQ1 downs 1.3-3.5% (the measured empty-launch floor is ~3 ns/block, 17.8 us for the 5760-block down launch). An LDS-staged activation variant was 4x slower: the per-lane 128-byte stride is an 8-way LDS bank conflict. |
| Output head and dense decode GEMV, attention prefill/decode, routed F16 prefill | Kept. The head streams 227-237 GB/s (near the 239.7 measured peak); the routed F16 decodes through LDS FP16-packed codebooks with XOR signs at fork-grade (22.8-26.5 TOPS, no slow-opcode pathologies in isa_mix); attention carries its own retained tuning history. |

### Prefill read pipelining with router predictions (2026-10-27)

Reads of a prefill layer's missing experts start during the previous
layer's compute: the previous layer's read overlap runs this layer's router
on its own FFN input (one layer stale), and the predicted experts' reads run
as a background job that the layer's prepare adopts for the experts it
routes ([design](STREAMING.md#prefetch-from-router-predictions)). This first
version gave background jobs a second whole-layer stage set in layer parity
(~3.6 GiB, bought after the seed), so the runtime cache grew to 98.0 instead
of 101.6 GiB.

A router-free sweep of the next layer's uncached experts, as ds4's Strix
Halo path does, was measured first and rejected on this model: V4.1 Flash routing is skewed enough that a
2048-token prompt touches only a fraction of each layer's experts, and with
the ranked seed resident the sweep read 3.6x the demand volume of the first
pass (22.9 vs 6.4 GiB), dropping pp2048 to 389.61/386.69 tok/s against
664.03/661.93 for the control. The predicted job reads ~1.6x the demand
volume, and the overlap absorbs it.

Matched `gufo bench` arms (pp2048/tg128, alternating binaries, targeted
shard page eviction between runs, two rounds each). The tg128 token hash
04842be7… and DSpark's 71 accepted of 107 drafted are identical on every
arm; the streamed pipeline changes only when bytes arrive, never which
bytes the kernels see.

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| control (028086ab) | 652.6–663.6 | 16.53–16.59 | 616.7–620.5 | 19.05–19.11 |
| prefill prediction | 779.3–785.1 | 15.75–15.94 | 737.2–748.0 | 17.78–17.91 |

Prefill gains 19–21% on both paths. Decode keeps its kernels and its host
wait (4.88–4.93 ms/token on every arm) but loses 4–6% to the smaller cache:
the second stage set is the price of the pipelining, which the compact
stage below recovers.

[Router-free sweep, candidate](artifacts/m3-streaming/bench-iq2-routersweep-cand1.log) ·
[Router-free sweep, control](artifacts/m3-streaming/bench-iq2-routersweep-ctrl1.log) ·
[Prefill prediction, AR](artifacts/m3-streaming/bench-iq2-prefetch-ar-cand1.log) ·
[Prefill prediction, AR control](artifacts/m3-streaming/bench-iq2-prefetch-ar-ctrl1.log) ·
[Prefill prediction, DSpark](artifacts/m3-streaming/bench-iq2-prefetch-ds-cand1.log) ·
[Prefill prediction, DSpark control](artifacts/m3-streaming/bench-iq2-prefetch-ds-ctrl1.log)

### Streamed decode prefetch (2026-10-27)

Single-token steps use the same pipeline: when a layer's
read overlap actually waited on reads, it enqueues the next layer's router
on the current FFN input, and the store thread submits the predicted
experts' reads, which the next layer's prepare adopts for the experts it
routes. Layers without misses predict nothing — a prediction without a
read stall to hide its matvec costs more than the read it saves — and
adoption skips unrouted experts, so wrong predictions cost bandwidth only.

Matched arms against the prefill-pipelining build (same protocol, two rounds; the
token hash 04842be7… and DSpark's 71/107 are identical on every arm):

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| prefill prediction only | 789.1–792.4 | 15.78–15.79 | 746.7–748.2 | 17.83–17.86 |
| with decode prediction | 778.7–788.0 | 15.91–15.92 | 748.4–750.2 | 17.84–17.89 |

Decode gains 0.8% on AR (every round separated well outside the ±0.01
run-to-run spread) and is neutral on DSpark, whose verification batches of
2–16 rows predict nothing by design. The gain is small because at 82–83%
route hits only ~5 ms/token of read stall exists and one-layer-stale recall
covers part of it; the larger decode lever was the cache capacity the second
stage set consumed (next section). Decode reads rise from 11.99 to
13.2 MiB/token (10.9 vs 4.1 GiB prefetched overall), the price of the
predictions that miss.

[Decode prediction, AR](artifacts/m3-streaming/bench-iq2-decode-ar-cand1.log) ·
[Decode prediction, AR control](artifacts/m3-streaming/bench-iq2-decode-ar-ctrl1.log) ·
[Decode prediction, DSpark](artifacts/m3-streaming/bench-iq2-decode-ds-cand1.log) ·
[Decode prediction, DSpark control](artifacts/m3-streaming/bench-iq2-decode-ds-ctrl1.log)

### Compact prefetch stage and cache reserve (2026-10-27)

The decode dip of the prefill pipelining came from the second stage set
taking ~3.6 GiB of expert-cache headroom; two changes give the memory back.

- **The prefetch stage no longer mirrors the demand stage.** Demand reads
  were always safe in one whole-layer buffer — the blocking routes copy
  orders them behind the previous layer's compute — so only the background
  jobs needed a second buffer, and jobs touch just the predicted experts.
  The prefetch stage is now a 192-row compact buffer with two 96-row parity
  halves (~1.5 GiB): a job's experts take rows in its layer's half, a job
  beyond 96 experts truncates (the excess stays a demand read), and a failed
  job is simply not adopted. Resolving the job moved ahead of the phase's
  table build so the table can point adopted experts at their rows.
- **The cache reserve drops from 8 GiB to 4 GiB.** The resident path runs
  with a 1 GiB dispatch reserve under the same loads; the streamed reserve
  now covers the shared batch scratch (1.4 GiB at 1M context), one session's
  FP4 state and transient peaks. 3 GiB measured one 572 tok/s prefill round
  riding the reclaim floor; 4 GiB is tight across every round.

Matched arms against the decode-prefetch build (same protocol). The cache reaches
103.8 GiB (AR) / 99.1 GiB (DSpark, which loads the drafter); hashes and
DSpark's 71/107 identical on every arm:

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| decode prefetch (8 GiB reserve, whole-layer set) | 787.5–788.0 | 15.86–15.92 | 744.5–749.3 | 17.87–17.91 |
| compact set, 4 GiB reserve | 739.9–747.9 | 17.34–17.38 | 763.9–765.1 | 19.03–19.07 |
| original control (028086ab) | 652.6–663.6 | 16.53–16.59 | 616.7–620.5 | 19.05–19.11 |

Decode recovers past the original control (+4.8% AR, parity on DSpark, from
4–6% below it), and DSpark prefill improves. AR prefill sits 5% under the
decode-prefetch build while DSpark prefill gains: across the pipelined builds the
AR prefill rate anti-correlates with the route-hit rate (88.3% hits → 787,
95.3% → 742 tok/s), so the residual is a residency effect (the Engram reads of the
[read-path profile](#storage-reads)), not a pipeline cost — reads fall from 13.2 to 9.3 MiB/token on the same
arms.

[Compact stage, AR](artifacts/m3-streaming/bench-iq2-compact-ar-cand1.log) ·
[Compact stage, DSpark](artifacts/m3-streaming/bench-iq2-compact-ds-cand1.log)

### Opt-in streaming, IQ1_M and the compact stage stride (2026-10-27)

Streaming became opt-in (`--stream-experts`, `--expert-cache-bytes`); a
model that does not fit now fails to load instead of streaming silently.
While streaming, the 5.31 GiB of dense weights go through the device heap
instead of managed memory.

**Compact stage stride.** The 3K teacher-forced score of the streamed
IQ2_XXS build had fallen from 6.0041 (`4b5ee8ea`) to 35.83 (`10ad16a4`).
Bisection put it on the compact prefetch stage (`522c8653`): disabling
prefetch or wide jobs restored 6.0041, and so did a whole-layer stage, also
with jobs still truncated at 96 experts. The stage is sized by the largest
expert of any layer, but rows were addressed with the current layer's
expert size; the build mixes 56-, 66- and 74-byte block types, so a layer
with large experts ran past the start of the other parity half, and the next
layer's job overwrote rows the current layer was still reading. Rows now use
the allocation stride. The streamed score is 6.0041 again, identical to
`4b5ee8ea` on all 2,999 positions. `ds41.expert_store` gained a mixed-size
fixture (odd layers half the size) that fails without the fix. `gufo bench`
token hashes did not catch it: their synthetic prompts score no text.

**IQ1_M streamed.** The IQ1_M build has 17 layers of IQ1_S experts and two
IQ2_S down projections, which the streamed batched path had never run.
IQ2_S has no FP16 tile kernel, so those rows go through the table-indexed
expert GEMV; the store also skips dropped routes (-1) as the resident
kernels do, instead of failing the batch. Teacher-forced 3K, 2048- and
952-token exact batches:

| IQ1_M path | Perplexity | Target top-1 | Agreement with resident |
| --- | ---: | ---: | ---: |
| Resident | 6.320 | 57.7% | — |
| Streamed, 32 GiB expert cache | 6.303 | 57.8% | 96.1% |

Single-token decode scores are bit-identical to resident on 256 positions;
the batched difference is the FP16 tile and GEMV rounding seen between the
resident batched and decode paths.

`gufo bench` pp2048/tg128 on IQ1_M, with the memory free once `gufo serve`
is ready at 262,144 tokens (no drafter):

| Expert cache | Free memory | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| Resident | 5.6 GiB | 1,212.3 ± 33.9 | 24.44 ± 0.13 |
| 64 GiB | 48.6 GiB | 616.2 ± 29.4 | 16.45 ± 3.19 |
| 32 GiB | 78.8 GiB | 437.2 ± 5.8 | 12.17 ± 2.34 |
| 16 GiB | 93.3 GiB | 374.4 ± 1.3 | 9.13 ± 1.23 |

Streamed IQ2_XXS against the pre-merge build with the same stride fix,
alternated twice: tg128 18.81/19.13 against 19.13/18.96 tok/s, no
measurable change from the heap placement. Prefill was inconclusive
(pp2048 853/927 against 735/953, each ± 300-460): the startup seed caches
the 10,240 ranked experts (82 GiB) of 15,360, so the first repetition reads
the rest (81% route hits) and the second hits on every route.

### Slots per expert size and the full built-in ranking (2026-10-27)

Two changes let a capped cache hold more experts; neither touches the
arithmetic.

- **One slot size per expert size.** Every slot was sized for the largest
  expert of any layer, but the builds mix block types: IQ1_M has six expert
  sizes (6.59-8.26 MiB, mean 7.11) and IQ2_XXS eight (7.38-9.76 MiB, mean
  8.20), so 14% and 16% of the cache went unused. Slots now come in one
  class per distinct size, each growing by 128 MiB chunks (512 MiB before,
  to bound what a class strands in a part-used chunk). Once the cache stops
  growing, an expert may take the smallest larger free slot or evict from
  its own or a larger class, never a smaller one.
- **The built-in ranking orders all 15,360 experts.** The import kept the
  first 10,240 of the source permutation, so a cache larger than 82 GiB
  (IQ2_XXS) started part empty and the first request read the rest. The
  table now keeps the whole ranking; the first 10,240 entries are
  unchanged. Experts a custom histogram omits follow its ranked ones,
  interleaved across layers.

`ds41.expert_store` gained cases for both (seven half-size experts in the
cache of four full-size ones, a large expert that streams rather than take
a small slot, and the seed past a one-expert histogram); they fail on the
previous store.

`gufo bench` pp2048/tg128 (`-r 2`), against the opt-in build above:

| Model, expert cache | Experts seeded | pp2048 | tg128 | Route hits |
| --- | ---: | ---: | ---: | ---: |
| IQ1_M, 64 GiB | 7,436 → 8,658 | 616.2 → 751.6 ± 98.1 | 16.45 → 17.90 | 70.8% → 85.7% |
| IQ1_M, 32 GiB | 3,718 → 4,325 | 437.2 → 462.5 ± 1.8 | 12.17 → 12.94 | 45.5% → 48.6% |
| IQ1_M, 16 GiB | 1,859 → 2,155 | 374.4 → 383.6 ± 2.0 | 9.13 → 9.74 | 30.2% → 32.7% |
| IQ2_XXS, automatic (105 GiB) | 10,240 → 12,688 | 853/927 → 918.9 ± 403.5 | 18.81/19.13 → 19.96 | 91.4% → 97.0% |

Free memory is unchanged: the cap bounds the allocation either way. The
seed takes longer in proportion (13.1 → 14.6 s at 64 GiB, 18.7 → 20.2 s
for IQ2_XXS). The IQ2_XXS prefill spread no longer comes from expert reads
(1.57 MiB/token in the first repetition, 93.6% hits); the Engram row reads
([read-path profile](#storage-reads)), whose page cache the full expert
cache displaces, are the likely remainder.

### Decode profile and per-layer expert tables (2026-10-27)

A rocprofv3 trace of warm IQ2_XXS decode (119 tokens of tg128 after
pp512, automatic cache, medians per token) splits the 50.1 ms token into
40.3 ms of kernels and 9.8 ms of GPU idle:

| Part | ms/token |
| --- | ---: |
| Dense Q5_K GEMV (`gemv_kernel<13,…>`) | 22.2 |
| Routed expert GEMV | 11.5 |
| Other kernels (attention, hyper-connections, router, …) | 6.6 |
| Idle: miss reads between the hit and miss launches (~4 per token) | 5.3 |
| Idle: ~200 small blocking copies | ~3 |

The dense GEMV runs near the measured bandwidth ceiling, so speculative
verification, which reads it once for several tokens, is the larger decode
lever. Timers around the Engram futures and the read phases put the Engram
wait at ~0.1 ms per pp2048 batch and the first prefill's expert read wait at
0.35 s (0 once warm): the Engram is not what slows a new prompt.

The copies were each layer's three expert tables and two hit/miss row
arrays, uploaded with blocking `hipMemcpy`. The store now keeps one device
table per layer and uploads it asynchronously only when one of its entries
changed; a warm cache almost never changes one. The hit/miss rows are
uploaded, in one asynchronous copy, only for a batch that both hits and
misses — the only case whose launches read them. The routes copy that
starts the next prepare orders every upload before its pinned mirror is
rewritten. Copies per token fell from ~200 to 23 (the slot fills after
reads), GPU idle from 9.8 to 8.1 ms and the traced token from 50.1 to
48.1 ms.

Alternated `gufo bench` (pp2048/tg128, `-r 2`): tg128 20.23/20.40 →
20.93/20.95 tok/s (+3.2%); pp2048 985.9/1061.2 → 1074.3/1077.2, inside its
first-repetition spread. Streamed IQ2_XXS decode scores are bit-identical
on 256 positions with the automatic cache and with a 16 GiB cache (71%
route hits, so most layers split).

### DSpark on streamed IQ2_XXS (2026-10-27)

`gufo bench` pp512/tg512 with the Lucebox Q2_K/Q4_K drafter: 33.5 tok/s
against 21.3 plain (+58%). Route hits settle at ~99.3% either way, but
verification reads are never predicted (0% prefetched), so ~13 MiB/token
of misses remain. A rocprofv3 trace of the second half of a DSpark tg512
splits the span into routed experts 44% (36% the 8-row verification
groups: ~36 distinct experts per layer at ~32 µs each, the single-token
per-expert cost, so bandwidth-bound), dense GEMV 31% (8-row launches at
1.4-1.8× the single-token time), other kernels 11%, idle 10% (about half
of it miss reads) and the drafter 4%.

The task-major multi-token dense GEMV ([resident engine
table](#resident-engine-m1m5)) took 8-row verification steps 129.7 →
118.8 ms on resident IQ1_M; streamed DSpark tg512 stayed inside its
cache-warming spread (33.17–34.14 → 33.15–34.73 tok/s over three
alternating pairs).

### E18: cost-aware draft prefix (retained)

Greedy cycles with the learned head now verify the draft prefix that
maximizes expected emitted tokens per cycle time, using the head's
predictions as acceptance probabilities and the session's measured cycle
times by width (the timing policy's estimates). Every emitted token still
takes exactly one draw at its position, so outputs do not depend on the
width. A cycle's time error also moves the other widths' estimates by 5%;
without that, widths the cut seldom picks kept their priors.

Per-cycle traces (all five drafts, the head's predictions and the decoded
sequence) give each draft's uncensored acceptance. The head is close to
calibrated (0.9+ rows accepted ≥ 99%, 0.5 rows about 55%). Verification
times by rows (`--spec-bench`) are 43.4, 58.0, 69.8, 78.8, 96.6, 107.5,
120.4 and 133.8 ms on streamed IQ2_XXS for 1-8 rows; 41.1, 52.8, 65.5,
75.7, 87.0, 98.3, 113.5 and 123.6 ms on resident IQ1_M. Replayed over five
prompts against the ≥ 0.5 cut, expected tokens per time gains +3.1%
streamed and +0.9% resident; a cumulative-confidence cut gains less
(+2.4%), any fixed width loses.

Matched A/B, five prompts × two alternating repetitions of 512 greedy
tokens against `d258afc1`:

| Model | Geomean | Faster | Base → candidate tok/s range |
| --- | --- | --- | --- |
| Streamed IQ2_XXS | +3.76% | 10/10 | 29.37-38.87 → 30.37-40.03 |
| Resident IQ1_M | +0.65% | 8/10 | 31.15-43.74 → 31.39-43.65 |

All 40 runs match plain decoding token for token. Fewer drafts are verified
(streamed 404-440 instead of 468-492 per 512 tokens) at higher acceptance.
The coding prompt, already at 84-88% acceptance with full blocks, measured
−1.1% and −0.2% resident. Rejected: the same cut with per-width estimates
only (+2.91% streamed, −1.87% resident, 0/10 faster): the 1- and 3-draft
estimates stayed at their 69.6 and 95.5 ms priors against about 62 and
86 ms measured, so those widths were almost never chosen. Runs, traces and
the replay are in [the E18 record](artifacts/2026-10-27-cost-aware-draft-prefix/summary.txt).

## Prefill at depth (2026-10-27)

Native pp2048 on resident IQ1_M fell from 1,126 tok/s at depth 0 to 1,017
at 64K and 838 at 128K. A rocprofv3 trace of 128K of depth preparation put
the growth in two kernels on encoder layers 2, 8 and 14, which score every
prompt row against depth / 2 compressed index keys: per 2,048-token chunk the
indexer scores grew 4 → 227 ms and the top-k 1 → 270 ms by ~115K, while
attention, projections and mHC stayed flat. Both kernels serve the streamed
`IQ2_XXS` model unchanged. [Record](artifacts/2026-10-27-indexer-depth/summary.txt).

| Depth | Base | E19 | E19 + E20 | E19–E21 |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 1,133 | 1,161 | 1,182 | 1,217 |
| 64K | 1,017 | 1,109 | 1,155 | 1,185 |
| 128K | 838 | 1,003 | 1,098 | 1,126 |

Depth 0 is the first test of each process and reads low in this harness
(1,126, 1,161, 980 and 1,122 across the four builds); a matched d0 A/B of
E19 + E20 against the base sat inside the runs' own spread.

### E19: two-pass indexer top-k (retained)

The radix select read each token's score row six times with per-thread
contiguous slices (uncoalesced) and histogrammed the top byte, where nearly
every score shares one of a few bins of LDS atomics. Now one coalesced pass
builds a 12-bit histogram (narrowed only while the k-th key's bin holds more
than 2,048 keys), one ordered pass appends the higher bins' indices and the
bin's keys to shared lists, the exact key and its ties are resolved in
shared memory, and two index-ordered lists are merged. The output keeps the
old order: keys above the k-th in index order, then the first ties (not one
sorted list; a first draft that sorted everything differed). Per index layer
at 128K: 102.3 → 4.7 ms; at depth 0, 0.29 → 0.48 ms. Byte-identical on the
128K/64K dumps and on 3,400 adversarial rows against the old kernel.

### E20: register-resident keys for the WMMA indexer scores (retained)

Each block covered 16 tokens and 128 compressed rows and converted every
head's FP32 queries again, about 16 GB of query reads per layer at 128K.
Now each of 16 waves keeps its 16 rows' keys as binary16 A fragments in
registers, a block covers 256 rows, the next head's queries are loaded
before the current head's products and stored after them, and the scores
leave through shared memory as contiguous rows. rocwmma's fragments had
held each 16-dim step's two halves swapped in the upper half-wave, which
changes the odd rows' summation order; the raw-builtin kernel mirrors that
(selecting in registers, so both half-waves keep reading the same LDS
addresses) and stays byte-identical. Per layer at 128K: 84.5 → 39.9 ms
(13.1 → 27.8 TFLOP/s); 2× at every depth. Rejected variants: 512-row blocks
(45.7 ms), two 16-token tiles per block (50.4), interleaved head pairs
(56.2, register pressure), head weights from LDS (42.5), ReLU as a
compare-select (no change), a second pre-swapped query copy in LDS for the
upper half-wave (40.9: twice the LDS reads), two 16-row tiles per wave with
8 or 16 waves (47.7, 44.3).

### E21: prefill attention staging (retained)

`attention_f16_kernel` (20 layers, ~207 ms per 2K batch at any depth past a
few thousand tokens) read each compressed row's index from global memory
right before that row's bytes, a chain of dependent loads per 16-row tile;
branched on window versus compressed rows once per row; and addressed rows
with 64-bit per-lane arithmetic. The token's indices now sit in shared
memory, the branch is taken once per tile, and rows are 32-bit offsets from
uniform bases (compressed rows stay below 2^24 at any supported context).
Per layer (2,048 tokens at 128K, `ds41_attention_test --bench`): 10.52 →
10.03 → 9.20 → 8.48 ms, and 12–13% at depth 0, ratio 1 and the 140-row
decoder replay. Outputs byte-identical to the base kernel on four shapes
(ratios 2, 1 and 0). Rejected: the two head groups of a token adjacent in
the grid (no change), the rows' scale bytes staged in shared memory
(10.3 ms), skipping unit softmax rescales (9.8).

Also measured on the shared FP16 projection kernels and rejected: dense
Q5_K activation fragments read twice from LDS instead of exchanged with
`permlanex16` (33 → 28 TFLOP/s, LDS-bound); the dense kernel without its
per-tile guard on full tiles (spills, as its comment warns); routed
activations prefetched a stage ahead in registers (IQ1_S and IQ2_XXS both
~5% slower).

### E22: the FFN's closing expand fused with the next state pass (retained)

Each batched block ended with `hc_expand_sum` (the six routed rows, the
shared output and the residual into the next state) and the next block began with a state pass that read that state
again for its attention mixes and collapse. `hc_state_pass` now also takes
the rows to sum, so a block whose successor runs on the same rows (no Engram
add or replay boundary between them) closes with the successor's attention
state pass. Per layer at 2,048 tokens (4,096 channels in the
microbenchmark): 3.40 → 2.71 ms, byte-identical state,
mixes and collapse (also 1,904 and 17 rows). With bounded replay most
batches run only the 20 encoder layers: pp2048 at d16384 1,225.6 → 1,229.1
tok/s (two alternating runs each), greedy text identical with and without
`--exact-prefill`.

Rejected on the indexer scores (E20's kernel, 39.9 ms per layer at 128K):
two or four heads' queries staged per barrier (48.7 / 44.5 ms), the head
weights loaded one head ahead (42.7), and, as an upper bound, the upper
half-wave's B operand without its swap (41.1, and not byte-identical). The
time is linear in heads: about 6 ms fixed (writing the scores, decoding the
keys) plus 1.05 ms per head (33 TFLOP/s in the head loop).

### E23: FP16 inputs for the dense FP16 projections (retained)

Measured per launch at 2,048 tokens (`tools/bench/ds41_dense_f16_tune.hip`,
whose `wo_a` shape was 8× too large before: it is 1,024 rows × 4,096 per
group, 8 groups): `wo_a` 5.50 ms (25 TFLOP/s) and the shared expert's
2,304-row gate and up 1.75 ms each (28) against 33–34 for `q_b`, `o_b`
and the shared down. A flat 8,192-row matrix over the same 4,096-wide input
takes 4.27 ms: the grouped input (268 MB of FP32, re-read by each 256-row
block) is what costs. Dropping the activation loads takes `wo_a` to 3.96 ms
and the per-superblock weight loads to 4.39.

The kernel already rounds its activations to FP16 as it stages them, so FP16
inputs give the same products. A first FP16-input build was 15–20% slower on
every shape: its two-entry `uint4` prefetch arrays went to scratch memory (80
bytes per lane, 38 scratch accesses per kernel). As `ext_vector_type` arrays
they stay in registers, and FP16 rows (half the registers) run four 64-value
stages ahead instead of two:

| Shape | FP32 input | FP16 input |
| --- | ---: | ---: |
| `q_b` 32,768 × 1,280 | 5.13 ms | 4.95 ms |
| `o_b` 5,120 × 8,192 | 5.04 | 4.66 |
| shared gate 2,304 × 5,120 | 1.72 | 1.44 |
| shared down 5,120 × 2,304 | 1.41 | 1.37 |
| `wo_a` 8 × 1,024 × 4,096 | 5.32 | 4.25 |

The producers now write FP16 where the dense kernel is the only reader: the
matrix-core attention stores its FP8-rounded output as FP16 rows for `wo_a`,
`wo_a` stores FP16 for `o_b`, and the shared expert's gate/up read the FP16
FFN input the routed FP16 kernels already convert (now converted before the
streamed store's overlap, which runs the shared expert). Every path is
bit-identical: `ds41.mmq_typed` checks the three FP16 modes against the FP32
launch and `ds41.attention` the FP16 attention output against the rounded
FP32 one; the 3,000-position batched score (`ds41_load_probe --batch
--score`) gives the same argmax and NLL at every position. pp2048 at d16384:
1,234.4 → 1,281.4 tok/s (+3.8%, two alternating runs each). Streamed
IQ2_XXS gives the same 3,000-position scores before and after (perplexity
6.004 against the oracle's 6.028, argmax agreement 95.6%) and batches 2,048
tokens in 4,252 instead of 4,901 ms.

Rejected on the way: 128- and 512-row dense blocks (every shape slower,
`wo_a` 6.23 / 5.88 ms), neighbouring lanes on neighbouring 16-byte chunks of
a token's activation row (no change), and Gemma 4's binary16 K-quant GEMM,
which runs these shapes 10–30% faster with FP16 inputs but sums the upper
half-wave's products in a different order.

Profile of the 128K chunk afterwards: GPU time per 2,048-token chunk 1,481
→ 1,380 ms; routed experts 488 ms (35%), dense FP16 projections 393 (28%),
attention 163, the state passes 106, indexer scores 61, MMQ (Q8_0 `q_a`/kv)
50, the BF16 matmuls 49.

Also measured and rejected: the routed FP16 kernel's activations prefetched
a stage ahead in vector registers (IQ1_S gate+up 21.6 → 22.3 ms, IQ2_XXS
25.0 → 24.9; its arrays never went to scratch). Ablations of that kernel
(`ds41_mmq_bench --f16 1`, 2,048 tokens): without the matrix-core products
gate+up takes 11.5 ms for both types, without the weight decode 20.5
(IQ1_S) and 18.9 (IQ2_XXS): the products and the rest barely overlap, and
IQ2_XXS's two table lookups per eight weights cost about a quarter. The BF16
matmul (`matmul_bf16_wmma_kernel`, three BF16 terms per activation, 19–24
TFLOP/s counting all three) with the next K step's operands loaded during
the current one: bit-identical, router 1.27 → 1.67 ms, and 1.46 with
LDS-only barriers.

### E24: half-wave permutes without register copies (retained)

The FP16 kernels' `v_permlanex16` swaps between half-waves passed
`fetch_inactive` and `bound_ctrl` false. The instruction's `old` operand is
then tied to the destination, so the compiler copied (or zeroed) a register
before every permute. With both flags set, the instruction reads its source
lanes directly and the copies go away. Every lane is active in these
kernels, so the results do not change. Dense FP16 projections got 4–5%
faster per shape, routed FP16 gate/up/down 0.5–3%, and the matrix-core
attention 8.47 → 8.34 ms at 2,048 tokens. The 3,000-position batched scores
are byte-identical for IQ1_M and streamed IQ2_XXS. pp2048 at d16384:
1,284.9 → 1,307.6 tok/s (+1.8%, two alternating runs each).

### E25: routed FP16 staging and LDS lookups (retained)

Ablations of the routed FP16 gate+up (`ds41_mmq_bench --f16 1`, 2,048
tokens, 32 or 48 slots per expert) split IQ1_M's 21.8 ms into the
matrix-core skeleton (14.7 ms: block fields, fragment reads, products), the
weight loads (+1.0), the activation staging and its two barriers per stage
(+1.7; +4.9 for IQ2_XXS) and the weight decode with its permutes (+4.4).
Making the codebook lookups conflict-free (every lane reading the same entry)
took IQ2_XXS from 24.5 to 20.4 ms and IQ1_M from 21.2 to 20.2: random
16-byte LDS rows cost about 2 ms per IQ2_XXS table.

Retained, each byte-identical on the 3,000-position batched scores of IQ1_M
and streamed IQ2_XXS:

- Double-buffered 64-value activation stages, the next one loaded before the
  current stage's products and stored after them, one barrier per stage:
  gate+up 21.84 → 21.09 ms (IQ1_M), 25.07 → 24.42 (IQ2_XXS), 21.50 → 20.79
  (IQ1_S), down 4–5% faster. 128-value buffers were neutral (51 KB of LDS
  took IQ1 from 6 to 4 waves per SIMD); 32-value stages gained 1.5% on IQ1
  and lost 2.5% on IQ2_XXS. IQ1_M pp2048 at d16384: 1,310.0 → 1,317.2 tok/s.
- IQ2 sign indices as eight bytes (0x80 per negative weight) placed by one
  `v_perm` per word instead of 16-byte FP16 masks: IQ2_XXS gate+up 24.46 →
  22.72 ms, IQ2_XS 24.82 → 23.21. IQ1 keeps the raw 4-byte codebook entries
  and maps their nibbles to FP16 high bytes with a `v_perm` lookup (half the
  table): gate+up unchanged, down 1–5% faster. Streamed IQ2_XXS pp2048 at
  d16384: 1,268.3 → 1,281.9 tok/s.
- For the IQ2 formats, each half-wave reads its own 16 bytes of an
  activation fragment and permutes the other half across lanes (the dense
  kernel's scheme): IQ2_XXS 22.82 → 21.93 ms, IQ2_XS 23.23 → 22.45. It costs
  the VALU-bound IQ1 formats 3–7%, which keep two LDS reads. Streamed
  IQ2_XXS pp2048 at d16384: 1,284.4 → 1,292.7 tok/s.

Rejected: the fragments of all four tiles read before the decode
(IQ1_M 21.2 → 25.8 ms: dead tiles' reads cost LDS bandwidth), and only live
tiles' fragments read before the decode (22.6). In the dense FP16 kernel,
two LDS reads in place of the fragment permutes were 25% slower on every
shape (it is bound by fragment reads), and waves of 32 rows × 64 tokens
(two B operands, half the fragment reads per product) were 5–10% slower
(238 registers, 6 instead of 8 waves per SIMD, twice the decode).

### E26: Engram staging, FP16 `q_b` rows and 128-row BF16 tiles (retained)

A 128K-depth profile after E25 (1,342 ms of GPU time per 2,048-token chunk)
left 18.8% of the wall span idle. Timing `eval_rows` showed 16 ms per chunk
before the first launch, spent allocating and zero-filling three host
vectors for the Engram rows (`n × 6,144` floats per Engram layer) that then
went up with synchronous pageable copies, draining the queue at each Engram
layer. The pinned Engram staging now holds a full batch (capacity rows), so
every batch's rows are read straight into it and uploaded asynchronously:
pp2048 at d16384 1,325.4 → 1,347.9 tok/s (+1.7%), byte-identical scores.
Between chunks only the embedding step remains (host dequantization and a
synchronous upload, 5–8 ms).

`q_kv_norm` also stores `qr` as FP16 rows (in the attention output buffer,
idle until the attention writes it) for the dense `q_b` projection, which
rounded them the same way when staging; the BF16 indexer projection keeps
the FP32 rows. `ds41.attention` checks the FP16 copy bit for bit. pp2048 at
d16384: 1,352.6 → 1,357.1 tok/s.

The batched BF16 matmul takes 128 weight rows per workgroup when the row
count allows (8 waves of 16 rows × 64 tokens), so each activation step is
split into its three BF16 terms once per 128 rows instead of 64. Every
output keeps the same products and sums (identical output hashes):

| Shape (2,048 tokens) | 64-row tiles | 128-row tiles |
| --- | ---: | ---: |
| router 384 × 5,120 | 1.25 ms | 0.98 ms |
| compressor 512 × 5,120 | 1.52 | 1.51 |
| compressor 1,024 × 5,120 | 2.70 | 2.54 |
| indexer `q_b` 4,096 × 1,280 | 2.56 | 2.21 |

In the model the BF16 matmuls of three d4096 chunks took 163 → 139 ms
(total GPU time −0.65%), below the end-to-end noise of pp2048.

Rejected: 32-token BF16 tiles (twice the workgroups, every shape 17–30%
slower), and skipping the attention output's softmax rescale when every
factor of a wave is exactly 1 (a uniform branch around the multiplies:
8.36 → 9.2 ms; interleaved with the products they cost nothing). Also
measured: the Engram projection is 25,600 × 6,144 (17.8 ms per call, 36
TFLOP/s; the tune bench had a 1,280-wide input), and FP16 rows would save
0.8 ms of it.

### E27: one joint verification for concurrent DSpark sessions (retained)

Concurrent sessions verified jointly in groups of up to 16 rows (E14), but
each still drafted a full block, so from four users a step needed two or
more joint evaluations and mixed text decoded slower than AR. A batch now
gives each session at most its even share of one 16-row evaluation
(16 / users − 1 copies or greedy drafts), plus the rows earlier sessions
left, and verifies anchors only past eight users. Cycles in a batch no
longer feed the batch's time into a session's own width timings. Sampled
drafts keep their own width: a shorter block consumes different draws, and
seeded outputs must not depend on the batch (the `batch` suite's
cross-endpoint replay caught this).

On the card workload (the same prompt for every user, greedy) mixed text
rises 42.6 / 43.7 / 44.7 → 51.4 / 55.0 / 57.8 tok/s at C4 / C6 / C8 and
repetitive text 75.4 / 74.9 / 74.8 → 89.2 / 97.4 / 92.3, every completion
identical to the isolated AR reference. With a different prompt per user
(11-prompt corpus) it rises 40.6 / 42.1 → 47.6 / 51.5 at C4 / C8, against
46.0 / 54.6 for AR.

Two policies on top were measured and not kept. Trimming drafts after
drafting by an expected-tokens-per-ms model (v2) stayed within noise. A
batch plan before drafting (v3), from measured drafter (8–10 ms per session,
run one after another) and joint pass costs, skips drafting where it does
not pay: +6–13% on the card's mixed text, −4–7% on its repetitive text, and
a tie on distinct prompts (47.3 / 52.0). The card's identical prompts keep
v1's sessions in lockstep, so a 16-row pass holds four copies of the same
tokens and shares their experts (166 against 185 ms once sessions drift);
that lockstep, not the policy, explains v1's repetitive lead. The remaining
cost is the serial drafter forwards ([backlog](BACKLOG.md) item 10).
Runs are in [the E27 record](artifacts/2026-10-29-joint-draft-sizing/summary.txt).

### E28: Engram row reads under a full expert cache (rejected)

BACKLOG item 11 blamed the Engram's ~98,000 random 4 KiB reads per pp2048
batch for streamed prefill slowing as the expert cache fills. On real text
a batch reads 49,152 rows per Engram layer, 39–43K of them unique (12–20%
repeats); unique 4 KiB pages are slightly more than unique rows (144 B rows
straddle pages), so sorting by offset shares nothing and deduplication
would remove at most ~15% of the reads. As an upper bound, a timing-only
build skipped every Engram read (`IQ2_XXS`, `--stream-experts`, synthetic
tokens, eight pp2048 tests per process, alternating twice): warm batches
1,365 → 1,376 tok/s (+0.8%), the first batch 1,027–1,063 → 1,064–1,102
(+3.6%). Streamed prefill now runs ~1,370 tok/s, against the 742–787 the
item was based on, and the Engram reads no longer limit it; a hot-row cache
would also take memory from the expert cache. Not pursued.

### E29: 6-column dense and grouped expert GEMVs (retained)

BACKLOG item 1 expected 4–5 ms from 5- and 6-row verifications running the
8-column kernels. The dense split GEMV and the grouped expert GEMV now
instantiate 6 columns for 5–6 rows (expert groups of 5–6 tokens). Every
row stays bit-identical: the kernel test matches 5-, 6- and 7-row launches
against one launch per row for every type, and `--spec-check` keeps widths
5 and 6 identical to single-token decode. `--spec-bench` (IQ1_M, 1,096-token
prompt, alternating builds, ms): 5 rows 89.5 / 89.4 → 89.1 / 88.9, 6 rows
100.5 / 100.6 → 99.6 / 99.9, other widths unchanged. The column jump the
item measured is gone on the current kernels (4 → 5 rows adds 12.4 ms, on
the per-row trend), so the gain is 0.5–0.8%.

### E30: expert-contiguous repack, read side (rejected)

BACKLOG item 12 proposed repacking the streamed build so each expert's
gate, up and down rows are one read instead of three about 1 GiB apart. A
read benchmark over the real `IQ2_XXS` expert spans (O_DIRECT, four
readers, per-layer batches, alternating arms; [script](artifacts/2026-10-29-expert-repack-io/repack_io.py))
compared the three projection spans with one contiguous span of the same
size per expert, and with one long run per layer:

| Batch | Three spans per expert | One span per expert | One run per layer |
| --- | ---: | ---: | ---: |
| 6 experts per layer (decode) | 5.53–5.83 GiB/s | 5.70–5.83 GiB/s | — |
| 64 experts per layer (prefill) | 4.66–4.68 GiB/s | 4.70–4.74 GiB/s | 6.35 GiB/s |

A ~2.7 MiB projection span already reads at the drive's large-IO rate, so
contiguity per expert is worth ~1%. Only long runs of neighbouring experts
gain (+36%); a rank-ordered file would give those only when a batch misses
many rank-adjacent experts, mainly a cold first prefill, while decode
misses a few random experts and warm prefill waits ~0 on reads. Not worth
a second model conversion for users.

### E31: next-layer prediction for verification batches (retained)

BACKLOG item 3: verification and joint batches (2–16 rows) predicted no
next-layer experts, so a streamed DSpark step read every missed expert on
demand. They now predict, like single tokens, when the current layer waited
on reads. Reads only; outputs are unchanged. `gufo bench` (`IQ2_XXS`,
`--stream-experts`, DSpark Q2K/Q4K, pp512 then six tg128 tests per process,
alternating, tok/s): the first test 45.21 / 43.71 → 46.77 / 46.66
(+3.5% / +6.7%), the second 50.52 / 52.26 → 49.73 / 50.82, warm tests 3–6
53.9 / 53.2 → 53.6 / 53.0 (within noise). The gain is while the cache still
misses; charging rows their own miss bytes stays open.

### E32: batched paths for 7- and 8-token prompt batches (retained)

BACKLOG item 8: small prompt batches were not monotonic in cost, because up
to 8 rows took the per-row decode paths (`gufo bench`, IQ1_M, ms): pp6 91.3,
pp7 107.7, pp8 121.8, pp9 ~98 once warm. Prompt evaluations need no
single-token arithmetic, so they now switch to the batched paths above 6
rows; decode, verification and joint evaluations keep theirs. pp7 107.7 →
~99, pp8 121.8 → ~99 ms (five repetitions, two alternations), pp2–pp6 and
pp9+ unchanged. The first batch of a new batched shape in a process costs
~20% more once, in both builds. `cache-edits` and `long-context` pass; the
DS4.1 GPU tests pass.

### E33: pinned embedding staging and threaded dequantization (retained)

BACKLOG item 7: each prefill batch dequantized its embedding rows on one
host thread and uploaded them, with the layer-0 one-hot pre-mix, through
synchronous pageable copies while the GPU waited. The rows and pre-mix now
stage in a pinned buffer beside the Engram staging and upload
asynchronously, and batches of 256 rows or more dequantize on four threads.
Embedding stage per 2,048-row batch (IQ1_M, pp8192, host timer up to a
device synchronize): ~6.0 → ~4.5 ms (dequantization 2.0 → 1.2 ms, upload
~2.5 → ~1 ms); the remaining ~2 ms are the device copies into the residual
streams. pp2048 / pp8192 throughput moves within noise (~0.1%). Greedy
output is identical, and `--joint-check 4` keeps all 164 rows bit-identical.
