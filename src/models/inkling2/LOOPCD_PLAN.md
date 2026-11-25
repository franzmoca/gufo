# LoopCD for Inkling 2 Mini: working plan

Branch `feat/inkling2-loopcd` (from upstream `main` 4060f397). Working notes, not
for upstream. Discussion: https://github.com/gufo-org/gufo/discussions/1407

## Goal

Opt-in LoopCD-Logits ([arXiv 2610.02185](https://arxiv.org/abs/2610.02185)) for
Inkling 2 Mini, then measure whether it improves quality enough to justify its
cost. Off by default; with it off nothing is allocated and output is
bit-identical to `main`.

Recipe (paper Eq. 3-4, the author's Inkling recipe, omlx #5640, SparkInkling):

- Comparison readout z_1: the state after pass 1 of the core (core output `cur`
  and pass 1's block states) runs the coda (layers 18-23) again in **its own
  cache slots** (5 sliding-window visits + 1 global, own conv state), then the
  depth mix, final norm, head and soft cap. Pass 1 has no messages and uses the
  physical slots 6-17, and the coda takes no messages, so z_1 equals the model
  with `n_passes = 1` bit for bit: the main correctness oracle.
- `z' = z + w (z - z_1)`, `w = w_max (1 - (p1 - p2))` from softmax(z), after
  the soft cap, in FP32. `w_max = 0.5` (the author's setting for Inkling; the
  paper's adaptive caps are 0.25-1.5 and stay positive over [0.5, 1.0]).
- The margin is taken from z before constraints and logit bias (omlx takes it
  after its vocabulary bias); document the choice.
- Modes: `off`, `on` (every row guided), `skip-thinking` (rows inside a thinking
  block unguided; on guided rows control-token probabilities stay exactly those
  of z and ordinary tokens are renormalized to the same total mass, as omlx).
  skip-thinking is the author's recommended recipe ("leave thinking blocks and
  control tokens untouched"), especially for coding agents; SparkInkling's
  measured gains come from `on` (reasoning loops ended). Both are evaluated
  from M2 on.
- Guided logits are no longer bounded by the cap: |z'| <= (1 + 2 w_max) * 30,
  i.e. 60 at 0.5. No current path assumes +-30 (the head GEMV caps once;
  `CoupledSample`'s softcap is on drafter logits), but assert it in a test.

Non-goals for now: LoopCD-Hidden, the read-only coda variant (no own cache),
fewer passes, per-request strength.

## 1. Options

- `--loopcd off|on|skip-thinking` and `--loopcd-strength W` (w_max, default
  0.5, range (0, 4]) in `extension.json` `options` (serve, bench, prompt),
  read in `provider.cpp` through `ext::FindOption`
  (`include/gufo/ext/options.hpp`). Names open; one option per behavior. No
  ext-api change needed.
- Per-request strength would need a request field the extension API does not
  have: out of scope, note it in the discussion if asked.
- Test-only: a way to run the engine with `n_passes = 1` (ParseConfig refuses
  it today through `c.Expect("loop.count", ...)`) for the oracle and the
  weak-model diagnostic (7.1). Keep it out of the CLI.

## 2. Engine state (the largest refactor)

The engine and the cache layer hard-code 48 visits in many places
(`state_layout.hpp` `kVisits`, slot table `fs::SlotEntry[48]`, `d_tbl + b*48`,
conv state `48 * kConvChannels * 3`, `d_vcap` `[48][6656][rows]`, `Tap`,
prefill visit units, `PrefillCheckpointScratchBytes`, cache components).

- Introduce a visit count = 48 + 6 reference visits when LoopCD is on, 48 when
  off. Replace the literal 48s with it (off keeps every size and bit).
- Placement: drafter rows sit at visits `kVisits + l` (48-50). Reference
  visits get a fixed base after them, `kVisits + kDraftLayers + l` (51-56),
  whether or not a drafter is loaded, so the drafter's numbering never depends
  on LoopCD and the `off` layout is unchanged. Engine cache slots for the
  reference coda are 48-53 (after the 24 + 24 core/coda slots).
- Memory per session:
  - global reference slot: 4 KiB/token (K+V f16). This is the cost that
    matters: 8 -> 9 global visits, +12.5% append KV per token, about 4 GiB
    for a 1M-token session. KV-pool admission and max-context checks must
    count it (SparkInkling's pool shrank 10-12%).
  - five window rings: 5 x `ring_rows()` (>= 1024 + 255) x 4 KiB, about 25
    MiB.
  - conv: +6 x 78 KiB.
  - rows mapped as the session grows, like the existing global slots.

## 3. Forward

Common to decode (`BuildDecode`, graph), batched decode (`DecodeN`), verify
(`VerifyN`, up to 64 rows / 16 batched) and prefill (`Ubatch`). Passes are
numbered from 1 below (code `pass == 0` is pass 1).

1. **Stash pass 1.**
   - Decode: block buffers come from the 16-entry `NewBlock` pool, reset once
     per step, so pass 1's blocks survive; keep their pointers and copy only
     `cur` (`d_bankout` is overwritten when pass 3 starts). Check the pool
     still fits once the reference coda adds its blocks.
   - Prefill: `blocks`/`bank` are swapped between passes, so pass 1's blocks
     are overwritten by pass 3: copy them (or give pass 1 separate buffers).
2. **Main coda** as today; DSpark `CaptureFeature` stays on the main path (the
   drafter keeps the unguided features).
3. **Reference coda**: layers 18-23 on the stash, reference slots, own conv
   offsets. Runs on every row, prefill included (its cache must be complete).
4. **Reference head**: `DMix(output_res_query)` over the reference states,
   `output_norm`, head GEMV with soft cap, into a second logits buffer. In
   prefill it runs on **every row whose logits leave the engine**: the last
   row, all rows when a `sink` is set, and interior checkpoint rows
   (`need_all`, `cp.logits` from #1403).
5. **LoopCD kernel**, in place on `logits_d` before argmax / linear sampling /
   readback (decode, verify and the prefill `chunk` readback): per row one
   sweep for top-2 + softmax denominator, one sweep to combine; per-row
   guidance flag from device memory (skip-thinking) so graphs stay valid;
   control-token renormalization (logsumexp over ordinary ids for z and z').
   Every sampling fast path (device argmax, GPU linear sampling, coupled
   proposals, verify) then reads guided logits with no runner change.

Implementation order: first sequential (steps 3-4 after the main coda, simple
and easy to check), then the **stacked** version: the reference rows join the
coda layers as extra rows with their own slot-table entries (like extra
sessions in `DecodeN`), and both head rows go through one GEMV call, so the
coda's dense weights and the head are read once. The head switches from
`GemvQ8` to `GemvMma` at `kMmaHeadRows = 8` and the MoE kernels have row
tiers, so going from nb to 2nb rows can change kernels: the stacked z_1 is
compared with `n_passes = 1` at the same kernel tier, or within a stated ULP
bound (6.2).

Rollback and checkpoints: rejected drafts must roll back the reference window
rings and conv tails like the main ones (`d_vcap` for 54 visits,
`CommitVerify`); the in-pass prefill checkpoints from #1403 must include the
reference slots.

## 4. Runner

- skip-thinking flag per row from the runner's `Block` state (`kThinking`).
  For verify rows, advance the block tracker over the draft tokens so each
  row gets the flag of its own position (drafts can cross `<|end_message|>`).
  The last prefill row takes the block state at the end of the prompt.
- Control-token id set from the tokenizer's `is_special`; check whether it is
  the contiguous range >= 199998 (`n_vocab_text` 200058) the other engines
  assume.
- Constraints (tool grammar) apply after LoopCD; nothing changes there.

## 5. Cache

- New components for the reference slots (append + window + conv) in
  `state_layout`; bump `StateMeta::kVersion`.
- Identity (memory and disk) includes LoopCD on/off: a prefix saved with
  LoopCD off has no reference cache.
- The private `kLogits` component (last-token logits) and the #1403
  checkpoint logits are **guided** logits, so they depend on mode and w_max.
  Either put mode and w_max in the identity of those components, or store
  the unguided z and z_1 and guide on restore. Start with the identity (simple);
  KV/conv components stay shareable across strengths only if the components
  are keyed separately.

## 6. Correctness

Every check has a pass criterion; a skip (missing model, missing GPU) is not a
pass. GPU checks use the `gpu-test` preset (assertions on), focused targets:
the existing `inkling2_spec_test`, `inkling2_cache_test`, `inkling2_oracle`, plus
a new `inkling2_loopcd_test`. Functional checks use `tests/functional/run.py`
with the Inkling devlab profiles (`inkling-2-mini/off`, `inkling-2-mini/dspark`).

### 6.1 Kernel (`inkling2_loopcd_test`, GPU, synthetic logits)

| Check | Criterion |
|---|---|
| Fixed w, adaptive w, skip mask, control renorm vs host FP64 oracle | max abs diff <= 1e-5 relative to row max; argmax equal |
| Top-2 with ties, w at margin 0 and 1, all-equal row | matches oracle |
| Control renorm: control-token probabilities vs z | equal to 1e-6; ordinary mass sums to the same total |
| One row alone == same row in a batch of 1/8/16/64 | bit-identical |
| w = 0 (any mode) | output == input, bit-identical |
| Guided logit bound | \|z'\| <= (1 + 2 w_max) cap |

### 6.2 Engine (GPU, real weights)

| Check | Criterion |
|---|---|
| `off`: existing `inkling2_*` tests | unchanged |
| `off`: greedy text of a fixed 20-prompt set (prose, code, tools, thinking, AR and DSpark, 1 and 4 sessions) | text hashes equal to `main` |
| `on`, w_max -> 0 path (strength test hook) | logits bit-identical to `off` |
| Reference logits (sequential) vs `n_passes = 1` engine: decode, `DecodeN`, verify, prefill (ubatch boundaries, interior checkpoints) | bit-identical |
| Reference logits (stacked) vs same | bit-identical at the same kernel tier; otherwise max abs diff <= stated bound, argmax equal |
| Graph == eager under LoopCD | bit-identical |
| Concurrent sessions == alone | bit-identical |
| `on` greedy text differs from `off` | yes on >= 18/20 prompts (sanity that it is active) |
| Prefill sink rows and checkpoint rows | guided logits equal the decode-path logits at the same position |

### 6.3 Speculative decoding (`inkling2_spec_test`)

| Check | Criterion |
|---|---|
| DSpark verify == AR, greedy, `on` and `skip-thinking`, 1 and 4 sessions | identical text |
| Same, sampled with a fixed seed | identical text |
| Forced rejection at every draft position, incl. drafts crossing `<\|end_message\|>` (skip-thinking flags) | identical to AR; reference rings/conv rolled back (state hash equal to AR) |

### 6.4 Cache (`inkling2_cache_test` + functional `cache`, `long-context`)

| Check | Criterion |
|---|---|
| Save/restore in RAM and disk round trip with LoopCD on | continuation identical to an uncached run |
| Off-cache into an on-server, and the reverse | not restored (miss), output identical to uncached |
| Restore across a w_max or mode change | not restored for the logits component (or re-guided), output identical to uncached |
| #1403 in-pass checkpoints with LoopCD on | identical to uncached |
| Window boundary (> 1024 + chunk rows) with LoopCD on | identical to uncached |

### 6.5 Options (`tests/cli/` style, CPU)

Parsing, defaults, range refusals (`0`, `> 4`, non-numeric), unknown mode,
`--loopcd-strength` without `--loopcd`.

### 6.6 Functional suites

The Inkling devlab suites (`src/models/inkling2/functional/devlab.json`) on
both profiles with `--loopcd on` and `--loopcd skip-thinking`; correctness
assertions must pass. Plus `pi_agent.py` replays (7.4), sessions retained and
tool results inspected.

## 7. Evaluation

Pre-registered before M2: arms, sets, settings, endpoints and decision rules
below are fixed before any LoopCD result is seen; changes are written down
with the reason, and every run is retained (raw requests, replies, token
counts, server logs), including void and negative runs.

### 7.1 Weak-model diagnostic (before M2, cheap)

Inkling is unlike the paper's four families (message-bank loop, MoE, attention
residuals). With the `n_passes = 1` path, on 20 fixed prompts (prose, code,
math, tools; thinking on), teacher-forced over the `off` greedy output:

- top-1 agreement of z_1 and z, mean KL(p || p_1), mean w, fraction of rows
  with w > 0.25, and top-1 flip rate of z' vs z (`on`, w_max 0.5 and 1.0);
- 1-pass greedy text for 3 prompts (is it coherent?).

Reported, not a gate, unless the flip rate is ~0 (nothing to measure) or z_1 is
incoherent while flips are frequent (then expect losses, say so in #1407).

### 7.2 Arms and settings

- Arms: `off`, `on`, `skip-thinking`, all at w_max 0.5. Second strength 1.0
  (not 0.3) only for an arm that passes the M2 gate.
- Binary: `result/bin/gufo` (or `build/release/gufo`) from the same revision for
  all arms; only the flag differs. Same weights (devlab asset, pinned
  revision), same server options, AR (no drafter) for quality runs: DSpark is
  exact (6.3), so it would change speed, not answers.
- Decoding: greedy (temperature 0) for the primary runs. Concurrency 8 is
  allowed only if 6.2 "concurrent == alone" passes; otherwise concurrency 1.
- Noise floor: the `off` arm is run twice on the M2 sets; any discordance
  between the two runs is reported and caps how much a LoopCD difference can
  be trusted.
- Thinking budget regimes (Inkling's default effort is `high`, no default
  budget):
  - **R1 no budget**: `max_tokens` 32768, no `reasoning_budget_tokens`. A reply cut
    at the cap counts as wrong. This shows the loop effect without budget
    forcing.
  - **R2 budget**: `reasoning_budget_tokens` 16384, `max_tokens` 20480. This is
    the fair test SparkInkling left open: does LoopCD add anything once
    budget forcing closes long thoughts?
- Harness: small external script (never `gufo eval`) talking to the OpenAI
  endpoint, reusing SparkInkling's/omlx's set definitions where possible so
  results are comparable; answer extraction and HumanEval execution
  (`he_exec.py`-style, sandboxed) identical across arms.

### 7.3 Sets

| Set | Items | Effort | Regimes | Role |
|---|---|---|---|---|
| MMLU-Pro, fixed subset (seed recorded, stratified by category) | 150 at M2, 500 at M4 | high | R1, R2 | **primary** |
| HumanEval (base + EvalPlus extended) | 164 | high, low | R1 | secondary |
| GSM8K, fixed subset | 150 | low | R1 | negative control (expected flat) |
| MMLU-Pro, same subset | 150 | low | R1 | negative control |
| Inkling functional suites | all in devlab.json | default | - | must pass (6.6) |
| `pi_agent.py` tasks | all prompts, 3 repeats | default | - | agent workflow (7.4) |

### 7.4 Agent workflow

`tests/functional/pi_agent.py` against a local server per arm, all prompts, 3
repeats each (sampled at the model's default temperature; seeds recorded).
Record per task: pass/fail, tool-call validity, number of turns, tokens,
wall time; inspect the actual tool results and keep the sessions. Report per
task, never only an aggregate.

### 7.5 Metrics and statistics

- Per item: correct, tokens (reasoning, answer), cut at cap, budget-forced
  close, loop watchdog firings, wall time.
- Paired comparison per set and regime: lost / gained items, exact two-sided
  McNemar (sign test on discordant pairs). Report the counts, not only p.
- Primary endpoint: MMLU-Pro high accuracy, R2 (budget), `on` vs `off` and
  `skip-thinking` vs `off`; Holm correction over the two arms.
- Secondary: MMLU-Pro high R1, HumanEval (both efforts, base and plus, also
  pooled), cut-at-cap counts, mean tokens, end-to-end suite wall time.
- Negative controls: a significant change (p < 0.05) on GSM8K low or MMLU-Pro
  low in either direction is reported and investigated before any claim.

### 7.6 Decision rules

- **M2 gate** (MMLU-Pro 150 high R1 + R2, HumanEval high/low, `on` and
  `skip-thinking`): continue with an arm if gained > lost on the primary set
  in R2, or in R1 with p < 0.1, and no secondary set has net loss >= 3 items.
  Stop the project if no arm meets it; publish the numbers in #1407 anyway.
- **Final claim** (M4: MMLU-Pro 500, full sets): propose upstream only for an arm
  whose primary endpoint is p < 0.05 after Holm, with no secondary or control
  set significantly worse; otherwise report as inconclusive or negative.
- The recommended default mode in the docs is the arm with the better primary
  result; ties go to `skip-thinking` (cheaper with DSpark, the author's recipe).

## 8. Speed

Method (AGENTS.md rules): `result/bin/gufo` / `build/release/gufo`, tuned
profile (accelerator-performance) and fan control checked first, no builds
during timing runs, alternate base/candidate (ABBA). Timings are compared per
request and per phase at 5% and 10 ms, never averaged across requests; noisy
results stay inconclusive and are rerun only for the affected histories.

| Metric | Arms | Gate |
|---|---|---|
| Decode tok/s AR and DSpark, prose and code, 1 and 4 sessions | main vs `off` | **no regression** (the 48 -> variable refactor) |
| Prefill tok/s and TTFT at 4K and 32K | main vs `off` | no regression |
| Same decode/prefill set | `off` vs `on` vs `skip-thinking`, sequential and stacked | reported; stacked must beat sequential |
| DSpark acceptance (accepted drafts per step) | `off`, `on`, `skip-thinking` | reported (SparkInkling: -10-13% with `on`, unchanged with skip-thinking) |
| KV-pool capacity (tokens) and max context at default memory | `off` vs `on` | reported; matches the 2. memory estimate within 5% |
| Device memory per session at 32K / 256K | `off` vs `on` | reported |
| End-to-end wall time of the 7.3 suites | all arms | reported (guided answers can be shorter) |

## Progress

- **M1 (complete)**:
  - Options `--loopcd off|on|skip-thinking` and `--loopcd-strength` (`0.5`
    default, `(0, 4]`) on `serve`/`bench`/`prompt`; `inkling2.probe` covers option
    validation and refusals.
  - Parameterized `slot_stride` / `n_slots` across HIP decode/attention/conv
    kernels; added `LoopCdGuide` (FP64 online top-2 + softmax sum, `guide_mask`,
    `control_bits` logsumexp renorm) with `inkling2_loopcd_test` §6.1 unit
    checks passing.
  - Sequential reference coda + head in `Engine` (`n_slots_=54`, `kRefVisitBase=51`,
    `kRefVisits=6`), `StateMeta::kVersion=3`, `kStateAbi="inkling2-hip-m2-v3"`, and
    `BuildIdentity` cache key isolation (`loopcd=off` vs `loopcd=mode:w_max`).
  - Verified on gfx1151:
    - §6.1 kernel unit tests & §6.2 engine checks (`w_max=0` == `off`, `z_1` ==
      `override_passes=1`, graph == eager, concurrent == alone, sink & interior
      checkpoints, 19/20 prompts differ, global KV ratio `1.1250` at 32K/256K):
      all pass.
    - §6.3 `inkling2_spec_test` (`off` 34.7% DSpark, `on` 25.2% DSpark) and server
      AR vs DSpark parity (`20/20` identical greedy outputs across `off`/`on` and
      `c1`/`c4`): pass.
    - §6.4 `inkling2_cache_test` (`off` 106 comps, `on` 119 comps, worst
      `max |dlogit| = 0.000e+00`, stale `kVersion=2` refused) and server RAM/disk
      cache hit + isolation (`on`/`off`/`w_max`): pass.
    - §6.5 `off` vs `main` (`4060f397`): `20/20` greedy server outputs identical;
      ABBA speed benchmark (`accelerator-performance`) shows `0` regressions
      (AR c1 `50.30–50.32` vs `50.26 tok/s`, Prefill 4K `2113.59` vs `2107.78`,
      Prefill 32K `2023.43` vs `2018.54`, DSpark c1 `75.99–76.13` vs `75.98–76.08`).
- **M1.5 (complete)**:
  - §4 special tokens: `vocab=200058`, `count=60`, strictly contiguous
    `[199998, 200057]`.
  - §7.1 diagnostic (4,434 rows): all `top1_agree=66.67%`, `mean_KL=0.9632`,
    `mean(1-margin)=0.2457`, `w_max=0.5` `mean_w=0.1229`, `flip_rate=8.23%`;
    thinking (3,863 rows) `top1_agree=65.00%`, `flip_rate=9.14%`; non-thinking
    (571 rows) `top1_agree=77.93%`, `flip_rate=2.10%`. 1-pass samples stay
    coherent initially then show repetition/arithmetic slips.
- **M2 (code & §6.2–§6.6 verified; quality gate pending)**:
  - Implemented per-row `guide_mask` & `control_bits` across `Engine`
    (`Forward`, `Decode`, `Step`, `DecodeBatch`, `DecodeBatchRows`, `Verify`,
    `VerifyBatch`), `Inkling2CacheAdapter::PrefillThrough`, and
    `Inkling2Runner` (`NextBlockSimple`, `PromptGuide`, `FillVerifyGuide`,
    `Tokenizer::IsSpecial` bitset).
  - `inkling2_loopcd_test`: `PASS §6.2 skip-thinking (guide=0 == off, ctrl_mass
    diff=9.82e-11, Verify == Decode bit-exact)`.
  - `inkling2_spec_test --loopcd skip-thinking`: `48/48` equal plain
    decoding, `30/119` (`25.2%`) DSpark acceptance, `k=1..6` boundary
    verification across `<|end_message|>` `0` errors.
  - `inkling2_cache_test --loopcd skip-thinking`: `worst max |dlogit| =
    0.000e+00`, `ALL PASS`.
  - Server verification (`build/release/gufo`): §6.3 `DSpark == AR` `20/20`
    greedy + `10/10` sampled (`seed=42`) for both `on` and `skip-thinking` at
    `c=1` and `c=4`; `skip-thinking` `reasoning_content` `100%` identical to
    `off` (`8/8`) while post-thinking answers differ (`4/8`); §6.4
    `skip-thinking` RAM (`2048`) & disk (`2247`) cache hits + `on`/`off` misses
    (`0`).
  - §6.6 functional suites with `--mmproj`: `stops`, `conversation`,
    `structured`, `tool-history`, `long-context`, and all `14` `cache` sub-runs
    (`baseline passed`) for both `--loopcd skip-thinking` and `--loopcd on`.

## Milestones

Each milestone ends with its exit criterion met and results retained.

1. **M1 (done)**: options, state refactor, sequential reference coda + kernel,
   `on` mode. Exit: 6.1, 6.2 (sequential), 6.3 greedy, 6.4, 6.5 pass; 8 "main vs
   `off`" has no regression.
2. **M1.5 (done)**: weak-model diagnostic 7.1.
3. **M2 (in progress)**: skip-thinking (runner flags, verify rows, control renorm)
   with 6.3 for it, then the quick quality signal for both arms. Exit: 7.6 M2 gate
   decided and posted.
4. **M3**: stacked coda/head. Exit: 6.2 stacked + 6.3 pass; speed table 8.
5. **M4**: full evaluation (7.3 at M4 sizes, 7.4, optional strength 1.0).
   Exit: 7.6 final claim decided; write-up in #1407.
6. **M5**: docs (`docs/models/inkling-2-mini.md`) and PR, or extension route,
   depending on the maintainers' answer.

## Open questions

- Upstream vs extension (discussion #1407, no replies yet).
- Does skip-thinking keep the gains that come from ending reasoning loops?
- Does LoopCD add anything once budget forcing is on (R2)?
- Exact cost of the stacked version on gfx1151 (expert overlap between the two
  coda rows).
