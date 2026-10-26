# DeepSeek V4.1 Flash benchmarks

`gufo bench` on the release build: IQ1_M resident (ISTA-DASLab GSQ-RCO,
112.43 GiB) with the IQ4_NL Engram tables read from disk; one Strix Halo
(gfx1151). Two repetitions per row; greedy decoding.

```sh
gufo bench -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf \
  -p 2048 -n 64 -d 0,8192,32768 -r 2
```

The 131K rows come from a separate `-d 131072 -r 2` run of the same build.

| Test | t/s |
|---|---:|
| pp2048 | 1176.72 ± 102.49 |
| tg64 | 23.99 ± 0.14 |
| pp2048 @ d8192 | 1196.32 ± 4.74 |
| tg64 @ d8192 | 23.47 ± 0.01 |
| pp2048 @ d32768 | 1121.00 ± 9.13 |
| tg64 @ d32768 | 22.95 ± 0.01 |
| pp2048 @ d131072 | 855.18 ± 0.25 |
| tg64 @ d131072 | 21.09 ± 0.03 |

Prefill runs the decoder layers (20-39) over a prompt's last 140 tokens
only (decoder SWA bounded replay, [QUALITY.md](QUALITY.md)); before it,
pp2048 was 703.07, @ d8192 677.76, @ d32768 585.30 and @ d131072 372.21
with the same decode, and `--exact-prefill` (`serve`, `bench`, `prompt`)
restores that path: pp2048 697.71 against 1144.10 in the same session of
runs. `gufo serve` prefilled an 11,327-token
prompt in 9.52 s instead of 16.95 s (1,190 tok/s).

`--prefill-batch 4096` (default 2048) prefills a 4096-token prompt at
720 tok/s instead of 694 (`gufo bench -p 4096 -r 3`) for about 1.2 GiB more
activation memory.

Prefill runs the routed experts, the dense Q5_K projections and attention on
the FP16 matrix cores (previous card: pp2048 421.80, @ d8192 414.37,
@ d32768 378.66; decode unchanged).

Prefill batches no longer shrink with the session context: the same
2,048-token prompt (`ds41_load_probe --batch`, first pass over the weights)
prefills at 384 tok/s with a prompt-sized context, 386 tok/s in a
131K-token context (was 245 with 409-token batches) and 381 tok/s in the
native 1M context (was ~120 with 51-token batches).

## DSpark speculative decoding

`ds41_load_probe --dspark-generate`, greedy, release kernels on the
`gpu-test` build; plain decoding of the same tokens as control. Output is
token-identical to plain decoding ([QUALITY.md](QUALITY.md)).

| Prompt | Drafter | Plain | DSpark | Accepted | Tokens/cycle |
|---|---|---:|---:|---:|---:|
| Roadmap document, 1,871 tokens, 256 generated | Q2_K/Q4_K | 23.46 | 28.49 | 66.4% | 2.46 |
| Roadmap document, 128 generated | Q2_K/Q4_K | 23.33 | 27.57 | 52.6% | 2.72 |
| Roadmap document, 128 generated | Q8_0/MXFP4 | 23.04 | 26.49 | 51.7% | 2.56 |
| Chat coding request, 45 tokens, 384 generated | Q2_K/Q4_K | 23.5 | 37.85 | 76.3% | 3.84 |
| Chat coding request, 384 generated | Q8_0/MXFP4 | 23.5 | 39.17 | 76.3% | 4.09 |

Prompt lookup in DSpark sessions: when the context repeats earlier tokens,
a cycle verifies up to six of the tokens that followed them instead of a
draft. On a file edit (rename a member throughout a 110-line header and
return the file; 1,302 tokens, 512 generated, Q2_K/Q4_K) DSpark alone
reaches 43.88 tok/s (5.17 tokens/cycle) and with lookup 56.43 tok/s (6.56
tokens/cycle; 71 of 78 cycles copied, 414 of 420 copies accepted), against
23.4 tok/s plain. The roadmap and chat prompts above rarely repeat
themselves (0 and 3 lookup cycles): 28.95 and 37.81 tok/s.

Without a drafter, decoding verifies prompt-lookup copies the same way: the
file edit runs at 54.61 tok/s against 23.59 plain (71 lookup cycles, 418 of
424 copies accepted, 7.21 tokens per cycle), and the chat prompt at 24.30
against 23.81 tok/s. A step emits the token after its accepted copies
without evaluating it (the next step's first row), so tokens without a copy
cost one evaluation each, as in plain decoding.

`gufo serve --dspark-model` (Q2_K/Q4_K, 32K context), 70-token thinking
prompt, 400 tokens: greedy 26.4 tok/s (155 of 277 drafts accepted),
temperature 1.0 23.6-24.6 tok/s (49-52% accepted), against tg64 23.88
without a drafter. The drafter adds 4.59 GiB (Q2_K/Q4_K) or 7.42 GiB
(Q8_0/MXFP4) of GTT.

DSpark sessions use the component cache; the drafter rings are one more
private component. With Q2_K/Q4_K, 32K context and one session, the cache
commits 1.5 GB of pinned backing and 1.8 GiB of host memory stays
available. A 27,559-token prompt prefilled at 568 tok/s; its capture took
9.5 ms (101 MB), and repeating the request restored it in 2.9 ms with the
same reply. Short turns capture in 0.9-4.2 ms (24 MB) and restore in
0.8 ms; whole-state snapshots took 14-16 ms for the same turns.

Host memory freed for the cache: the drafter's F16 Markov tables stay F16
(4.71 → 4.59 GiB, bit-identical drafts), and the DeepSeek cache keeps 16
transfer streams instead of 64 (each HIP stream holds about 9 MiB of host
memory; 0.38 GiB less resident).

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
