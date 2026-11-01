# HIP quantized matrix kernels

These model-private kernels derive from llama.cpp commit
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT, The ggml authors),
originally under `ggml/src/ggml-cuda/`. The DS4 adapter also carries changes
from `xangel82/DS4-GB10-GX10-DSpark-CUDA`, commit `910501e`.

Gufo builds this directory with HIP for AMD `gfx1151`. Device headers use
`.hip.hpp`; translation units use `.hip.cpp`. The adapter calls HIP/hipBLAS
directly and uses `ds41_ggml_hip_*` symbols to remain separate from Qwen's
matrix backend. There is no CUDA build or CUDA-to-HIP API alias layer.

`ds41_mmq.h` exposes the raw-pointer C ABI. `ds41_ggml_stubs.h` supplies the
small ggml type and allocation interface needed by the templates. Quantization
layouts, tables and active AMD arithmetic remain from the imported kernels.
Unused NVIDIA/MUSA preprocessing paths, ggml graph support and full ggml
tensor entry points are removed; the V4.1 decoder owns its HIP graph
execution.

When importing upstream changes, retain the HIP implementation and rerun
`ds41.mmq_typed`, which checks every quantized type the V4.1 builds use
against a matvec reference. Current measurements belong in the model's
[BENCHMARKS.md](../../../../../../docs/models/deepseek-v4.1-flash/BENCHMARKS.md).
