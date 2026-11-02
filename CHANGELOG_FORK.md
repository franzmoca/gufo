# Fork changelog

Changes made in this fork's DeepSeek V4.1 Flash and Gemma 4 engines, newest
first. Each entry is a brief summary; its pull request has the details,
measurements and validation. Upstream Gufo's changes are in
[CHANGELOG.md](CHANGELOG.md).

## 2026-11-03: DeepSeek V4.1 Flash: expert cache sized after the sessions ([#17](https://github.com/franzmoca/gufo/pull/17))

- On streamed builds, `gufo serve` now sizes the expert cache after it has
  created its sessions and the RAM prompt cache, leaving 2 GiB free. Before,
  the cache was sized at load and everything allocated later had to fit in
  a fixed 4 GiB. With one session at the native 1M context, the
  `UD-IQ2_XXS` expert cache grows from 95.5 to 97.4 GiB. Decode is about 3%
  faster and SSD reads per decoded token drop 6-11%. With four sessions the
  cache shrinks to 94.6 GiB and the server stops swapping (8 GB swapped
  before). Greedy outputs are unchanged.
- The automatic RAM prompt cache of a streamed model is 1 GiB with
  `--cache-disk` and 3 GiB without, instead of half the free memory.
  `--cache-ram-bytes` still sets it explicitly.
- The benchmark card adds `UD-IQ2_XXS` streamed: 16.8-18.8 tok/s AR and up
  to 61 tok/s DSpark for a single user, 76.3 tok/s AR and 93.4 tok/s DSpark
  for eight users, 17.1 s cold load.

## 2026-11-03: DeepSeek V4.1 Flash: Unsloth UD-IQ2_XXS and faster streamed prefill ([#16](https://github.com/franzmoca/gufo/pull/16))

- Streamed builds run Unsloth's `UD-IQ2_XXS`: five shards, with the ISTA
  `IQ4_NL` Engram linked as shard 6. New IQ3_XXS, Q2_K and IQ2_S expert
  kernels cover decode, batched verification and prefill. Its 18.5K-token
  perplexity is 5.71, against 6.27 for `IQ1_M`.
- Streamed prefill reads experts ahead and computes while they arrive. Cached
  experts run while misses stream from the SSD. Batches of 17-1,023 rows,
  and each batch's first layer, predict their experts, and Engram rows are
  read one batch ahead. `gufo bench` pp2048 on `UD-IQ2_XXS` gains 4-5%, and
  a fresh 24K-token document gets its first token about 2 s sooner.
- Served prompts replay the decoder layers once per prompt instead of once
  per cache checkpoint: 2.4-2.7 s off the first token of a 14-24K-token
  document. Checkpoints record where their decoder rows start. Checkpoints
  saved by older builds are not restored.
- Fixed nondeterministic cold first requests: a batch could evict an expert
  it had just routed as a cache hit. Eviction now pops from short
  per-size-class victim lists instead of scanning every slot.
- Decode: greedy DSpark drafts only when drafts save time (chat, code and
  document decode +3-9%), sampled top-p takes its softmax totals from the
  device (+2-4%), short passes use a narrow BF16 router (pp20-64 +6-7%),
  and the Q8_0 `q_b` GEMV is split four ways.
- `PREFILL.md` and `DECODE.md` profile served prefill and decode against
  the gfx1151 ceilings, and `BACKLOG.md` ranks the open work.
  `tools/ds41/evict_sim.py` and `route_occupancy.py` replay expert traces.

## 2026-11-01: DeepSeek V4.1 Flash: split expert reads across two drives and DSpark drafter fixes ([#15](https://github.com/franzmoca/gufo/pull/15))

- Streamed builds (`--stream-experts`) can read every expert miss from two
  NVMe drives at once. `tools/ds41/pack_expert_tails.py` copies the second
  half of each routed expert projection into a tails file (61.6 GiB for
  `IQ2_XXS`) for the second drive. With `MODEL.gguf.expert-tails` beside the
  first shard, each read is split between the GGUF and the tails file. A
  mismatched file is detected at load and ignored. Outputs are unchanged.
- On a Samsung 990 PRO plus a KIOXIA EXCERIA BASIC, a single expert miss
  takes 0.87 ms instead of 1.32–1.47 ms. `gufo bench` on streamed `IQ2_XXS`
  gains about 7.6% decode and 16% prefill. In served chat, document and
  coding-agent workloads, prefill is 2–13% and decode 0.5–6% faster at the
  full expert cache, rising to 47–65% and 12–23% with a 64 GiB cache.
  Usage is in `STREAMING.md`.
- The expert read pool has 8 workers instead of 4, and the expert-cache log
  lines report the interval's tokens and split host wait into route and read
  waits.
- `DS41_EXPERT_TRACE=path` records expert routes, cache hits and read
  timings. `tools/ds41/prefetch_sim.py`, `rank_eval.py` and
  `prefetch_fit.py` replay those traces to compare cache and prefetch
  policies.
- Unsloth's DSpark drafters now load their confidence head
  (`confidence_head.proj.weight`), which was ignored, so they draft with
  learned widths.
- The DSpark drafter no longer blocks the host after each feature injection,
  stages its inputs in pinned memory and runs its F32 router with a wave per
  row and token (131 → 36 µs per layer). Drafting takes 8.5 → 8.3 ms per
  cycle; outputs are identical.

## 2026-11-01: Upstream v0.28.1, ROCm 10.1 and streamed DSpark fix

- The fork is now based on upstream Gufo v0.28.1, up from v0.23.0, and builds
  with ROCm 10.1, which upstream uses from v0.26. Upstream's changes in
  between are in [CHANGELOG.md](CHANGELOG.md). Greedy outputs of DeepSeek V4.1
  Flash `IQ1_M` and Gemma 4 31B are unchanged, and `bench` speeds match the
  previous build.
- The fork's code is now split by model: `feat/ds41` (DeepSeek V4.1 Flash) and
  `feat/gemma` (Gemma 4) are model branches on upstream, and `main` merges
  both.
- ROCm 10.1's compiler (LLVM 24) made DeepSeek V4.1 prefill about 7% slower
  by generating 16-bit register code in the routed and dense FP16 GEMMs.
  Those two kernels now build without it: prefill GPU time drops 6.3%, back
  to the ROCm 7 level, with identical outputs.
- Requests with `logprobs` (new upstream) return them for every generated
  token, including tokens accepted from DSpark drafts on DeepSeek V4.1 and
  from MTP drafts on Gemma 4.
- Fixed a hang or crash at the first DSpark request on streamed builds
  (`--stream-experts`), caused by two threads stopping the expert prefetch at
  once. On `UD-IQ1_M` it hit 3 of 9 server starts before the fix and none of
  10 after; DSpark runs at 34.5–36.2 tok/s.

## 2026-10-31: DeepSeek V4.1 Flash: Unsloth UD quants and faster prefill and decode ([#14](https://github.com/franzmoca/gufo/pull/14))

- Unsloth's Dynamic `UD-IQ1_M` (130.2 GiB) runs with `--stream-experts`: F32
  routers, Q6_K dense weights and IQ2_S routed experts load, and a build can
  link ISTA-DASLab's Engram file as its shard 5 instead of downloading the
  same tables again. `UD-IQ1_S` uses only supported weight types but is
  untested.
- Faster dense prefill on the FP16 matrix cores, now including Q8_0:
  streamed pp2048 1,135 → 1,278 tok/s on `UD-IQ1_M` and 1,189 → 1,223 on
  `IQ2_XXS`. Q8_0 activations are rounded to FP16 instead of Q8_1 blocks;
  3K-token perplexity shifts stay within noise.
- Faster decode with outputs unchanged: lighter streamed expert loads, the
  shared expert and hyper-connection mixes overlapped with other work, and
  multi-token Q6_K/Q8_0 GEMVs on the task-major kernel (+4.6% at 8 users on
  `UD-IQ1_M`).
- The benchmark card is refreshed for all tables and adds `UD-IQ1_M`; its
  repetitive workload introduces the copied passage as filler text.
  README_DEEPSEEK.md lists the Unsloth builds and links the card instead of
  repeating its tables.

## 2026-10-31: DeepSeek V4.1 Flash: MXFP8 and Q5_K Engram tables ([#13](https://github.com/franzmoca/gufo/pull/13))

- The loader reads the MXFP8 (202.8 GB, the checkpoint's own format) and
  Q5_K Engram tables that Unsloth publishes beside IQ4_NL.
- [EVAL.md](docs/models/deepseek-v4.1-flash/EVAL.md) compares IQ4_NL with
  MXFP8 on `IQ1_M` and streamed `IQ2_XXS`: perplexity and MMLU-Pro,
  HumanEval+ and MBPP+ scores do not change significantly.

## 2026-10-29: DeepSeek V4.1 Flash: streamed IQ2_XXS, llama.cpp GGUF layout and faster decode ([#12](https://github.com/franzmoca/gufo/pull/12))

- `--stream-experts` keeps the dense weights of the 128.75 GiB `IQ2_XXS` build
  resident and streams its routed experts from SSD into a memory-bounded
  expert cache (`--expert-cache-bytes`). Streamed outputs are bit-identical to
  the staged build. With a Samsung 990 PRO, AR decode is 20.8–22.7 tok/s
  across depths and the server is ready in 22.5 s from a cold page cache.
- **Breaking:** only ISTA-DASLab's republished six-shard llama.cpp layout
  loads, with the Engram file linked as shard 6 and the image routing bias
  read from the backbone. Older four-shard files convert in place with
  `convert_ds41_upstream.py`.
- **Breaking:** streaming is opt-in. A model that does not fit memory fails
  to load and states what it needs, unless `--stream-experts` is passed.
- Faster decode, prefill and DSpark. On `IQ1_M` resident at depth 0:
  - AR decode 23.3 → 24.5 tok/s.
  - pp2048 1,077 → 1,228 tok/s.
  - DSpark 28.0 → 30.7 tok/s on mixed text and 65.3 → 73.6 on repetitive
    text.
  - At 8 users, AR 78.5 → 86.1 tok/s and DSpark mixed 55.9 → 73.8.

  Greedy and seeded outputs are unchanged by the speedups.
- The benchmark card adds `IQ2_XXS` streamed tables, and the fork README now
  leads with DeepSeek V4.1 Flash.

## 2026-10-27: DeepSeek V4.1 Flash: small explicit prompt caches and memory warnings ([#11](https://github.com/franzmoca/gufo/pull/11))

- An explicit `--cache-ram-bytes` up to the automatic budget is always
  accepted. With the DSpark drafter at a 256K context, a 512 MiB cache used to
  fail the load and now leaves 2.7 GiB of host memory free.
- The server warns at load when the weights do not fit the GPU heap, naming
  the kernel parameters that raise the limit, and when more than 1 GiB of
  CMA is free.
- README_DEEPSEEK.md adds a rootless Podman setup that needs no Nix on the
  host.

## 2026-10-26: DeepSeek V4.1 Flash engine ([#10](https://github.com/franzmoca/gufo/pull/10))

- A native DeepSeek V4.1 Flash engine keeps the 763B model's GSQ-RCO `IQ1_M`
  backbone resident on one 128 GB Strix Halo at the native 1M context and
  reads the Engram tables from SSD.
- DSpark speculative decoding with a separate drafter, prompt lookup of up
  to 15 tokens, and joint verification of concurrent sessions; greedy output
  is identical to plain decoding.
- Image input through the vision tower (`--mmproj`), V4.1 DSML tool calls,
  and RAM and disk prompt caching.
- `feat/ds41` became the fork's default branch; it contains the Gemma 4
  engine as well.

## 2026-10-24: Gemma 4: Upstream v0.23.0 and faster small binary16 GEMMs ([#9](https://github.com/franzmoca/gufo/pull/9))

- Merged upstream Gufo v0.22.0–v0.23.0. Repeated `--served-model-name` works
  for Gemma models.
- The vision encoder and binary16 prefill layers use wide GEMM tiles from
  128 rows instead of 1024. A 70-token image's prompt time drops from 104 to
  97 ms on the 26B-A4B; 31B UD-Q8_K_XL pp768 is +2.3%. Results are
  bit-identical.

## 2026-10-23: Gemma 4: Faster UD-Q8_K_XL prefill and MTP decoding ([#8](https://github.com/franzmoca/gufo/pull/8))

- Q8_0 projections verify MTP drafts on the WMMA kernel over the grouped
  weight layout. 31B UD-Q8_K_XL MTP decode is +2.2% on mixed text, +19–20%
  on repetitive text and +10–23% with two users; the 26B-A4B gains up to
  +2.6%. Results are unchanged.
- The 31B's Q8_0 attention output and MLP down projections get their own
  prefill GEMM plan, and GeGLU runs inside the Q8_0 gate/up prefill GEMM.
  31B UD-Q8_K_XL pp2048 rises from ~565 to ~610 tok/s; 26B-A4B prefill is
  +1.0–1.1%. Results are bit-identical.
- `--prefill-weights decoded` no longer copies Q8_0 weights: with GeGLU fused
  in the default path, the copies were slower and cost 32.4 GiB.
- The 26B-A4B's Q8_0 attention-output prefill runs on Gemma's dense GEMM
  (+0.7% pp2048 on UD-Q4_K_XL and UD-Q8_K_XL), and Q6_K prefill projections
  from 4096 rows take 144-token tiles (+1.3% pp2048 on an all-Q6_K 31B quant,
  +0.5% on a Q6_K 26B-A4B quant). Results are bit-identical.
- Sampled MTP on the dense 31B draws its drafts without the request's top-p,
  so the drafter's runner-up can be verified beside each draft; output still
  follows the request's sampling exactly. 31B UD-Q8_K_XL sampled decode is
  +1.8–3.2%; the 26B-A4B keeps top-p.

## 2026-10-23: Gemma 4: Faster sampled MTP verification ([#7](https://github.com/franzmoca/gufo/pull/7))

- Sampled MTP cycles verify from each row's exact top logits, selected on the
  GPU, instead of copying every row's full logits to the host. Output is
  bit-identical. Sampled MTP decode is +1.0% on 26B-A4B UD-Q4_K_XL (+0.9% with
  a repeat penalty) and +0.2% on 31B UD-Q4_K_XL.
- Synced with upstream main through gufo-org/gufo#729 (0.21.0).

## 2026-10-21: Gemma 4: Faster MTP decoding and opt-in prefill weight copies ([#6](https://github.com/franzmoca/gufo/pull/6))

- With an MTP drafter loaded, the target's projection weights are reordered
  at load into 128-row groups that the verification kernel reads as one
  stream. It uses the same bytes and gives the same results, at about 0.3 s
  more load time. MTP decode tg128 is +3.4% on 31B UD-Q4_K_XL and QAT and
  +1.0% on 26B QAT, with prefill unchanged.
- `--prefill-weights off|compact|decoded` (serve, prompt and bench; default
  `off`) keeps load-time prefill copies of the dense projections beside the
  GGUF weights. On 31B UD-Q4_K_XL pp2048, `compact` is +3.3% for +18.9 GiB
  and `decoded` is +6.5% for +43 GiB (QAT +3.6%, UD-Q8_K_XL +3.8%). On the
  26B-A4B, both modes copy the expert gate/up weights: +1.0% for +9.0 GiB.
  Prefill results are bit-identical in every mode.
- `gemma4.target`'s between-lookahead-stops check now compares against a
  matching decode row, so it passes on 26B QAT.
- Synced with upstream main through gufo-org/gufo#686 (0.19.0).

## 2026-10-20: Gemma 4: Faster MTP attention at depth and a per-cycle draft trace ([#5](https://github.com/franzmoca/gufo/pull/5))

- MTP verification attention folds its reductions into DPP additions and
  coalesces score reductions, and 9–12 verified rows launch as one block:
  31B five-to-sixteen-row verification attention 4–6% faster at 32K and
  64K; 31B MTP decode at 16K +7.2% (native tg128), ~31K HTTP requests
  +0.1–1.9%.
- Drafter step prices are remeasured on the drafters actually used (31B
  Q8_0 assistant, 26B Unsloth drafter), so the draft policy's price error
  at ~31K falls from 1.19 to 1.06 (31B) and from 0.76 to 1.04 (26B);
  throughput is unchanged.
- `--draft-trace PATH` (serve, prompt and bench) writes one JSON line per
  MTP verification cycle: rows, siblings, drafter steps, acceptance, the
  policy's estimates and stop decision, per-row target statistics and stage
  timings. `gemma4_draft_replay` checks and replays the draft policy offline
  from these traces.
- A study of the calibrated draft policy on exactly labelled chains found it
  accurate and near the best achievable before verification; no policy
  change.
- Synced with upstream main through gufo-org/gufo#665 (0.18.0).

## 2026-10-20: Gemma 4: MTP on depth-upscaled 31B targets ([#4](https://github.com/franzmoca/gufo/pull/4))

- MTP decoding accepts targets of up to 96 layers instead of 64, so
  depth-upscaled 31B models such as BeaverAI/Spectre-37B-v1a (72 layers) run
  with the 31B drafter instead of failing at the first verification cycle:
  greedy decode 9.8 to 18.0 tok/s on Spectre-37B-v1a Q4_K_M. 26B-A4B and 31B
  are unchanged.

## 2026-10-19: Gemma 4: Continuations from the component cache ([#3](https://github.com/franzmoca/gufo/pull/3))

- Gemma 4 serves continuations through upstream's component cache instead of
  whole-session snapshots. Checkpoints keep global rows, sliding-window rows
  and a small private state apart, so a conversation's checkpoints share rows
  and a restore loads only what its slot lacks: a 26B swipe of a 16K-token
  prompt restores 1 MB in under 1 ms.
- The disk cache (`--cache-disk`) uses the component disk tier: after a
  restart, a 5.5K-token 31B conversation reaches its next first token in
  0.43 s instead of 9.9 s.
- Prefill no longer splits at checkpoints, captures no longer wait for
  another session's prefill (26B: 8–9 ms against legacy's 204–325 ms), and
  two requests finishing prefill together decode as one batch.
- Fixed 31B restores stalling 4–17 ms after the cache grew its host memory,
  and captures stalling while it grew.
- Synced with upstream main through gufo-org/gufo#477 (0.17.1).
- `cache_snapshot_ms` reports a prompt capture's own time; it included the
  wait until the scheduler collected the finished capture.
- On 31B MTP, 1,105 of 1,164 per-request timing measures match the legacy
  cache, and prompt captures take a median 0.58 of legacy's time. Restores
  into a free slot, where legacy reused the other slot's session, remain
  slower.

## 2026-10-18: Gemma 4: Prompt lookup from upstream and fewer request stalls ([#2](https://github.com/franzmoca/gufo/pull/2))

- Synced with upstream main through gufo-org/gufo#613. Gemma 4 takes its
  prompt-lookup rules where they measured faster, and copies only what every
  comparable earlier passage agrees on: log extraction +26–28% on 26B-A4B and
  +15% on 31B, edits inside long contexts +2–3%.
- A request waiting for its prompt checkpoint no longer waits for another
  request's whole generation (a 1.2 s request took up to 20 s; now 1.7 s).
- Prompt-lookup copies are reported in `lookup_n` / `lookup_n_accepted` and
  the `/metrics` counters.

## 2026-10-17: Gemma 4: Faster prefill at depth and MTP verification ([#1](https://github.com/franzmoca/gufo/pull/1))

- Prefill attention and projections: 31B prefill +3.5% with no context, +9%
  at 32K and +22% at 128K; 26B-A4B +12–17% from 16K.
- MTP verification of two to four rows on one wave per head, re-priced draft
  costs and fewer decode round-trips: 31B MTP decode +3.5–9%, 26B up to +13%
  at depth.
- Synced with upstream v0.17.0. A vision sidecar beside the model is now
  prepared at load: the first image request is faster, and text-only serving
  loads about 0.7 s slower and holds about 2.4 GiB more.
- Tool-call fixes: argument schemas beside arrays without `items`, and
  bounded whitespace before a required tool call.
- Benchmark cards refreshed on October 17; the 31B QAT card uses a reduced
  grid.
