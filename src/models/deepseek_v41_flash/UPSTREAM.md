# DeepSeek V4.1 Flash sources

V4.1 semantics (CSA2 KV/index sharing, YoCo layout, shifted mHC, Engram,
sqrtsoftplus routing, the DSpark drafter) follow the official
DeepSeek-V4.1-Flash-1017 `inference/model.py` and `inference/engram.py`, and
the GGUF layout follows the `deepseek41` architecture of the ISTA-DASLab
GSQ-RCO builds. The independent CPU oracle lives in `tools/ds41/`. Status and
history:
[docs/models/deepseek-v4.1-flash/EXPERIMENTS.md](../../../docs/models/deepseek-v4.1-flash/EXPERIMENTS.md).

## Origin

The package started as a copy of Gufo's `src/models/deepseek_v4_flash`
engine, which is adapted from DS4 commit
`84cc882352757baf628a1776badf7cc54d584e28` ([LICENSE.ds4](LICENSE.ds4)). The
V4 graph, session and kernel layer have since been replaced and removed; the
GGUF loader (`runtime/model_data.cpp`), the tokenizer (`runtime/tokenizer.cpp`)
and the resident weight upload (`kernels/rocm/backend.hip.cpp`) still derive
from it. The V4 engine itself stays unchanged.

## Layout

- `engine.hpp` and `engine.cpp` are the Gufo-owned C++20 API that product code
  uses for model loading, tokenization, request sessions, snapshots, logits
  and cancellation.
- `runtime/` holds the V4.1 decoder (`v41_decode.cpp`), request sessions and
  the DSpark policy (`v41_session.cpp`), the drafter (`v41_dspark.cpp`), SSD
  expert streaming (`expert_store.cpp`), GGUF loading and the tokenizer.
- `kernels/rocm/` holds every V4.1 kernel (`ds41_v41.h` lists them) and the
  vendored llama.cpp quantized matrix kernels in `mmq/`
  ([VENDOR.md](kernels/rocm/mmq/VENDOR.md)).
- `vision/` holds the ViT encoder and image prompt construction; its BF16
  projections use Qwen's hipBLASLt GEMM wrapper
  (`src/models/qwen/hip/ops/gemm.hpp`).
- `continuation_adapter.cpp` connects the decoder state to the component
  continuation cache.

DeepSeek kernels are private to this package: a change here must not alter
another model's output or performance.
