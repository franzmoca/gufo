# Gemma / DeepSeek allocation-only comparison on ROCm 10.1

2026-11-04, Strix Halo gfx1151, Linux 7.2.11. **No current restore benefit
was demonstrated by the 2 MiB mapping isolation in these workloads.** The
historical multi-millisecond stall did not reproduce. This does not establish
that ROCm alone fixed it.

The user identifies the original Gemma measurement's runtime as **ROCm 7.3.x**.
The exact historical binary/runtime revision is not available here. This is a
current-runtime on/off comparison, not a matched 7.3.x versus 10.1 experiment.

## Build and experiment

[manifest.json](manifest.json) records source `main` at
`01fb6f4e39464003efffd97d79607005146280f0`, binary hashes and observed ROCr
mapping `/nix/store/88jjsz...-rocm-runtime-10.1/lib/libhsa-runtime64.so.1`.
Gemma model sources are identical to `feat/gemma` `83b764b4`.
Both binaries use the same pinned Nix environment and production `release`
CMake preset. B is the unmodified aligned implementation. A applies only
[unaligned-control.patch](unaligned-control.patch): exact-size ordinary
anonymous mappings without 2 MiB isolation, including the corresponding release
extent. It preserves page population, THP opt-out, page pinning, HIP registration,
`FD_CLOEXEC`, model code and all other behavior. The active fork is unchanged.

[measure.py](measure.py) owns isolated loopback servers, checks the expected
response code and complete output, and uses the existing functional metrics
recorder/comparator (5% and 10 ms, per request/phase). Commands, every response,
measurements and logs are retained in each run directory. The diagnostic
[registration_trace.c](registration_trace.c) records actual `hipHostRegister`
calls without changing arguments. It logs only registrations, not hot-path
restores. These are instrumented cache investigations, not standard uninstrumented
speed-card qualification.

Primary modes match the serving scripts' drafting backends:

- Gemma 4 31B UD-Q4_K_XL, Q8_0 MTP, request-scoped calibration, 24 GiB RAM cache.
- Full DeepSeek V4.1 GSQ-RCO IQ2_XXS, Q2_K/Q4_K DSpark, streamed experts with a
  fixed 48 GiB expert cache, 8 GiB continuation RAM cache. This is not the
  expert-pruned Coder or resident IQ1_M configuration.

Both use two sessions, context 16384, greedy decoding, thinking off and
**unlimited memlock**, matching the scripts. The logs verify actual long-term
page pins. ABBA order per model: A1 unaligned, B1 aligned, B2 aligned, A2
unaligned. Each run has 17 requests: four growing turns, unchanged retry,
seven distinct prompts, three older-prompt replays and two uncached controls.
The older replays must prefill zero tokens; cold controls must prefill everything.
Response codes differ between prompts, so restoring another prompt's state
cannot pass by returning the same fixed answer.

## Results

All 136 primary requests passed those output and cache-work checks. Candidate
runs match A1's outputs, input/output/cache tokens, prefill chunks, restore bytes,
snapshot bytes and drafting work. The unchanged DeepSeek A2 control differs by
one proposed draft on `capture_7` (14 versus 15); outputs, accepted tokens and
cache work still match. Raw summaries: [summary.json](summary.json).

| Model | A1 drafted / accepted | B1 | B2 | A2 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 31B MTP | 470 / 218 | 470 / 218 | 470 / 218 | 470 / 218 |
| DeepSeek V4.1 DSpark | 233 / 225 | 233 / 225 | 233 / 225 | 232 / 225 |

Actual post-readiness backing growth occurred in every run:

| Model | Segments at readiness → final | Bytes per segment |
| --- | ---: | ---: |
| Gemma 31B | 17 → 124 | 133,562,368 |
| DeepSeek V4.1 | 17 → 20 | 133,496,832 |

Every B cache-backing segment is 2 MiB-aligned; A cache-backing segments are not
all aligned. The table filters by the backing segment size; other HIP host
registrations are retained in the trace and excluded from those counts. Registrations
succeed in both variants. The whole-block mapping adds no change to logical
cache budgets or transferred byte counts.

Restore milliseconds, per request, without averaging different histories:

| Model / request | A1 | B1 | B2 | A2 |
| --- | ---: | ---: | ---: | ---: |
| gemma31b growing_1 | 1.402 | 1.074 | 1.361 | 0.968 |
| gemma31b growing_2 | 1.088 | 1.316 | 1.092 | 1.157 |
| gemma31b growing_3 | 0.826 | 1.135 | 1.279 | 1.266 |
| gemma31b unchanged | 0.839 | 1.288 | 1.144 | 0.804 |
| gemma31b restore_5 | 10.921 | 10.609 | 10.905 | 10.842 |
| gemma31b restore_6 | 10.928 | 10.726 | 10.782 | 10.913 |
| gemma31b restore_7 | 11.183 | 11.012 | 11.255 | 11.371 |
| ds41 unchanged | 0.834 | 0.861 | 0.825 | 0.767 |
| ds41 restore_5 | 0.835 | 0.847 | 0.849 | 0.848 |
| ds41 restore_6 | 0.838 | 0.902 | 0.883 | 0.864 |
| ds41 restore_7 | 0.846 | 0.857 | 0.894 | 0.848 |

Gemma same-slot restores transfer 1,111,056 bytes; older-prompt restores transfer
980,014,860–1,031,573,256 bytes. DeepSeek restores transfer 13,924,636–14,026,552 bytes; its growing
turns reuse live state without a restore. Exact byte counts remain per request.
No cache-restore or cache-snapshot timing comparison exceeds the standard gates.

**Whole-inference timing remains inconclusive.** Gemma's cold-prefill/decode
flags drifted as the host reached about 98 °C; the unchanged A2 control also has
flags. These do not support an end-to-end speedup or regression claim. DeepSeek
was restarted with cooldown to at most 65 °C before each run, with per-request
temperatures retained. B1 and A2 still have some non-cache timing flags; B2
passes. All flags remain in the individual comparison files. No margins were
widened and no noisy request was discarded.

## Historical-setting control

A short paired Gemma MTP history also ran with **8 MiB memlock**, verifying
zero long-term backing pins. It retains the four growing turns, unchanged retry
and uncached final-turn control (12 additional successful requests). Outputs and
cache work match, drafting is exercised, and the standard comparison passes.
This covers the same-slot case from the historical failed-pin report under the
current runtime. These are one paired observation per history, not an ABBA
performance qualification.

| Request | Unaligned ms | Aligned ms |
| --- | ---: | ---: |
| growing_1 | 1.376 | 0.914 |
| growing_2 | 1.222 | 1.196 |
| growing_3 | 0.929 | 1.261 |
| unchanged | 1.125 | 1.025 |

No multi-millisecond restore stall reproduced in this control either. See
[low-memlock-summary.json](low-memlock-summary.json) and both `gemma31b-8MiB-*`
directories.

## Retained unsuccessful attempts

`failed-gemma31b-8GiB` retains the first diagnostic run. Its older replay reused
2744 of 2751 tokens, then prefills the seven-token assistant suffix, rather than
meeting this experiment's strict zero-prefill requirement. The ledger was at
7.73 GB under its 8 GiB budget. The paired experiment was restarted with a
24 GiB Gemma cache; the complete older-prompt copies then remain available.
The failed attempt is not counted as a quality pass.

`interrupted-ds41-startup-hot` was stopped before any inference measurement to
cool the host after the Gemma investigation. It is not a completed run.

## Interpretation and decision

Keep P6 **deferred**: current Gemma/DeepSeek results do not establish a performance
case for upstreaming it. Leave the existing fork implementation intact; this
bounded investigation does not prove it unnecessary for every workload or host.

A runtime fix is consistent with the observations, but not established. Other
conditions also differ: the historical Gemma note describes 1,049,821,184-byte
segments, while the current runs use approximately 127 MiB segments. The current
population code starts parallel workers only when `bytes / 128 MiB >= 2`;
these smaller segments populate serially, whereas a roughly 1 GiB segment takes
the multi-worker path. This changes the thread-stack mapping/unmapping history
involved in the old failure. Kernel/glibc, cache implementation and pinning are
additional potential differences. A causal ROCm conclusion requires matching
those conditions and replaying against the old runtime.

These text-cache tests cover the stated quants, modes and request histories.
They do not qualify images, disk restore, AR, other quantizations or a full model
quality/speed sweep. Model kernels and numerical contracts were not changed.
