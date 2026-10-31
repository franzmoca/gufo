# DeepSeek V4.1 Flash Upstream

This package is a fork of Gufo's `src/models/deepseek_v4_flash` engine, which
stays unchanged. V4.1 semantics (CSA2 KV/index sharing, YoCo layout, shifted
mHC, Engram, sqrtsoftplus routing) follow the official DeepSeek-V4.1-Flash-1017
`inference/model.py` and `inference/engram.py`, and the GGUF layout follows the
`deepseek41` architecture of the ISTA-DASLab GSQ-RCO builds. The independent
CPU oracle lives in `tools/ds41/`. Status and history:
[docs/models/deepseek-v4.1-flash/EXPERIMENTS.md](../../../docs/models/deepseek-v4.1-flash/EXPERIMENTS.md).

The notes below describe the V4 engine this package was forked from.

## DeepSeek V4 Flash origin

The model-private DeepSeek V4 Flash engine is adapted from DS4 commit
`84cc882352757baf628a1776badf7cc54d584e28`.

Only the model graph, GGUF/tokenizer support, request-session state, and ROCm
kernel dependency closure are retained. DS4 command-line, HTTP server, agent,
evaluation, and disk-cache frontends are intentionally excluded so Gufo owns
those product surfaces.

The imported backend was converted to native ROCm/HIP source names and APIs.
It does not call Qwen kernels or share Qwen tensor layouts, dispatch policies,
or mutable state.

## Integration Boundary

`engine.hpp` and `engine.cpp` are the stable Gufo-owned C++20 API. Product
code uses that API for model loading, tokenization, request sessions,
snapshots, logits, and cancellation.

The imported implementation is part of the Gufo model package rather than a
nested external project:

- `runtime` owns independent C++20 modules for GGUF/model data, tokenizer,
  sampling, request state, snapshots, and ROCm graph execution.
- `kernels/rocm` owns every DeepSeek numerical kernel and ROCm dispatch rule.
- `engine.hpp` and `engine.cpp` directly own the native runtime handles.

The model package intentionally duplicates numerical code instead of calling
Qwen kernels. A DeepSeek kernel change must not alter another model's output or
performance.

The standalone distributed, tensor-parallel, SSD weight streaming, multi-GPU
placement, CPU reference graph, MTP, steering, and DS4 frontend
implementations are not built or shipped. The production runtime has one
single-device resident ROCm route and one authoritative state layout.

## Update Policy

When updating the imported engine:

1. Record the new immutable upstream commit.
2. Import only code exercised by the single-device ROCm route for the
   supported artifact.
3. Preserve the ROCm-only source and naming contract.
4. Reapply model-private changes inside this directory, never in Qwen code.
5. Run the DeepSeek model tests, the fast hardware suite, and the canonical
   Nix gate before recording new performance.
