# DeepSeek V4.1 Flash on one Strix Halo

Gufo runs **DeepSeek-V4.1-Flash-1017**, a 763B-parameter mixture-of-experts
model with a native 1M-token context, on a single AMD Strix Halo with 128 GB
of unified memory. The whole quantized backbone stays resident in GPU memory;
only the Engram lookup tables are read from the SSD, one row per token.

> [!WARNING]
> **Headless, dedicated deployment only.** The resident model leaves about
> 1.75 GiB of host memory free once the server is ready (native context,
> one session, DSpark drafter). A graphical desktop, a browser or any other
> GPU workload on the same machine competes for that memory and can push
> the server or the system out of memory. Run it on a machine that does
> nothing else, without a desktop session, and reach it over the network.

## What it runs

- **Backbone:** ISTA-DASLab's GSQ-RCO `IQ1_M` build (1.75 bits per weight),
  resident in GTT.
- **Engram tables:** the shared `IQ4_NL` file, read from disk.
- **DSpark speculative decoding** with a separate drafter, plus prompt lookup:
  copies of earlier context are verified up to 15 tokens at a time.
  Greedy output is identical to plain decoding.
- **Image input** through the DeepSeek V4.1 vision tower (opt-in `--mmproj`).
- Gufo's serving features: every text endpoint, RAM and disk prompt caching, and
  concurrent sessions verified jointly.

Benchmarks: [docs/models/deepseek-v4.1-flash/BENCHMARKS.md](docs/models/deepseek-v4.1-flash/BENCHMARKS.md).
Quality against an independent reference implementation:
[QUALITY.md](docs/models/deepseek-v4.1-flash/QUALITY.md).

## Memory

GTT and host memory with the recommended files, measured with `gufo serve`:

| Component | Memory |
| --- | ---: |
| `IQ1_M` backbone, resident | 112.3 GiB |
| One session at the native 1M context | +2.3 GiB (0.9 GiB state, 1.4 GiB shared batch scratch) |
| Each further session at 1M context | +0.9 GiB |
| DSpark drafter, Q2_K/Q4_K | +4.59 GiB |
| Vision sidecar (`--mmproj`) | +0.90 GiB |
| Engram tables | 103 GiB on disk, not resident |
| **Host memory available** (1M context, one session, DSpark, server ready) | **1.75 GiB** |

The prompt cache sizes itself from what is left; in this configuration it
holds about 1.25 GB of checkpoints. The machine therefore has no room for
anything else: no desktop, no second model, no larger prefill batches
(`--prefill-batch 4096` leaves the prompt cache no memory at all).

### Host setup

- **Hardware:** AMD Strix Halo (`gfx1151`) with 128 GB, Linux. Set the UMA
  carve-out (dedicated VRAM) in the firmware to its minimum; the model lives
  in GTT.
- **Kernel parameters** that let the GPU map nearly all memory:

  ```text
  amdgpu.gttsize=126976 ttm.pages_limit=32505856
  ```

- **Check that the limit applies:** `cat /sys/class/drm/card*/device/mem_info_gtt_total`
  should report about 133 GB. With AMD's DKMS driver the TTM module is
  `amdttm`, so the parameter is `amdttm.pages_limit`. If the GPU heap is
  smaller than the weights, the server warns that it places them in managed
  memory; the resident build does not fit that way.
- **No free CMA:** `grep CmaFree /proc/meminfo` should be near 0. Free CMA
  counts as available memory, but GPU memory cannot use it. Some kernels
  (Ubuntu 26.04) reserve several GiB as the KHO scratch area; boot with
  `kho=off` to return it. The server warns when it finds more than 1 GiB.
- **Storage:** about 231 GB for the backbone and Engram files, on an NVMe
  SSD. The Engram tables are read during every prefill and decode step.
- **Headless:** boot to a text console (`systemctl set-default
  multi-user.target`) and serve over the network.
- Optional: `ulimit -l unlimited` (or `--ulimit memlock=-1` in a container)
  lets the prompt cache pin its backing memory.

## Download the models

All files go into **one directory**: Gufo opens the other backbone shards and
the Engram file from the directory of the first shard.

```sh
pip install -U "huggingface_hub[cli]"
mkdir -p ds41 && cd ds41

# Backbone (4 shards, 120.7 GB) and Engram tables (110.6 GB)
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ1_M*" "*Engram*" --local-dir .

# DSpark drafter (recommended: Q2_K/Q4_K, 4.9 GB)
hf download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF \
  DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf --local-dir .
```

| Source | File | Notes |
| --- | --- | --- |
| [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | `*-IQ1_M-0000{1..4}-of-00004.gguf`, `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | The only build that fits resident. `IQ2_XXS` (138 GB) needs SSD expert streaming, which is planned. |
| [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) | `DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf` | **Recommended drafter**: 4.59 GiB resident. |
| [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | Supported: 7.42 GiB resident, about 2.6 GiB less host memory left, similar speed and acceptance. |

Use the smaller Q2_K/Q4_K drafter: in matched runs the Q8_0 drafter
accepted as many drafts at similar speed, and the memory it takes comes out
of the prompt cache. Other published drafters (unsloth BF16, Lucebox
MXFP4/Q8) were not measured.

## Patch the vision tower

Image input needs unsloth's `mmproj-BF16.gguf` plus each MoE gate's image
routing bias (`bias_vl`, 40 vectors of 384 FP32), which no published GGUF
carries: DeepSeek V4.1 selects experts for image positions with a separate
bias. `tools/ds41/vision_sidecar.py` copies the mmproj and appends those 40
vectors, read from the official checkpoint
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
with HTTP range requests (only those tensors are downloaded):

```sh
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF mmproj-BF16.gguf --local-dir .

# from the Gufo checkout; the script needs numpy (available in `nix develop`)
nix develop -c python3 -I tools/ds41/vision_sidecar.py \
  --mmproj ds41/mmproj-BF16.gguf --out ds41/mmproj-ds41-BF16.gguf
```

Pass `--checkpoint DIR` to read the bias from a local snapshot of the
official checkpoint instead. The server rejects the unpatched
`mmproj-BF16.gguf` with a message pointing to this script; pass the written
`mmproj-ds41-BF16.gguf`. Vision is opt-in: the server loads
it only with an explicit `--mmproj` (0.90 GiB), never by discovery.

## Build and serve

Build as described in the [README](README.md#build-from-source), then:

```sh
./result/bin/gufo serve llm \
  -m ds41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf \
  --speculative dspark \
  --dspark-model ds41/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf \
  --host 0.0.0.0 --port 8080
```

Add `--mmproj ds41/mmproj-ds41-BF16.gguf` for image input. The server is
ready in about 19 s from a cold page cache; the model stays resident until the
server exits. Useful options:

- `--context N` per session; the default is the native 1M tokens.
- `--sessions N` for concurrent users; each adds about 0.9 GiB at 1M context.
  Concurrent DSpark requests are verified in one joint batch.
- `--think off|on|auto` sets the default reasoning mode for requests that do
  not choose one.
- `--cache-ram-bytes N` caps the RAM prompt cache. With DSpark at a 256K
  context, `--cache-ram-bytes 536870912` leaves 2.7 GiB of host memory free
  instead of 1.9 GiB, for fewer cached conversations.
- `--cache-disk DIR` keeps prompt-cache checkpoints on disk across restarts:
  a 12K-token conversation restored after a restart in 13 ms (110 ms to the
  first token instead of 12 s). The first start with it hashes the weights
  (about 50 s); later starts reuse the hashes.

## Run in rootless Podman, without Nix on the host

Nix can run inside the official `nixos/nix` image instead. A named volume
holds its store, so the build survives the container and the server runs
from the same volume. The host only needs Podman and the GPU driver (with
the host setup above).

```sh
# Once: a volume for the Nix store, and the source
podman volume create gufo-nix
git clone -b feat/ds41 https://github.com/franzmoca/gufo.git

# Build (the first run downloads the toolchain into the volume)
podman run --rm -v gufo-nix:/nix -v "$PWD/gufo:/src:Z" \
  docker.io/nixos/nix:latest \
  nix --extra-experimental-features 'nix-command flakes' \
  --accept-flake-config build /src --out-link /nix/var/gufo

# Serve, with the models in ./ds41
podman run -d --rm --name gufo-ds41 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 \
  -v gufo-nix:/nix:ro \
  -v "$PWD/ds41:/models:ro" \
  docker.io/nixos/nix:latest \
  /nix/var/gufo/bin/gufo serve llm \
  -m /models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf \
  --speculative dspark \
  --dspark-model /models/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf \
  --host 0.0.0.0 --port 8080
podman logs -f gufo-ds41
```

- To update, `git -C gufo pull` and rerun the build command.
- `--accept-flake-config` uses Gufo's binary cache, so most of the toolchain
  is downloaded instead of compiled.
- `:Z` lets SELinux hosts such as Fedora write to the source directory; the
  Git reader locks it. It changes nothing elsewhere.
- `--device` passes the GPU; `--group-add keep-groups` keeps your `render`
  and `video` groups, which rootless Podman needs to open them. If ROCm
  finds no device, see the GPU access checks in the
  [Quickstart](README.md#quickstart).
- `--ulimit memlock=-1` lets the prompt cache pin memory, as on the host.
- The container saves no memory and needs no memory limit: in matched runs
  it left the same free memory as the native server. The kernel and its
  parameters are the host's, so the host setup above still applies.
- Stop it with `podman stop -t 30 gufo-ds41`.

## Performance

Measured over HTTP with `gufo serve` (greedy, thinking off, real chat
text); full tables and charts are on the
[benchmark card](docs/models/deepseek-v4.1-flash/BENCHMARKS.md).

| Workload | Depth 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| Prefill, pp2048 (tok/s) | 1,077 | 993 | 700 |
| Decode, AR (tok/s) | 23.3 | 22.9 | 21.7 |
| Decode, DSpark, mixed text (tok/s) | 28.0 | 30.1 | 27.2 |
| Decode, DSpark, repetitive text (tok/s) | 65.3 | 59.6 | 55.2 |

| Concurrent users | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| AR, total decode (tok/s) | 23.8 | 39.6 | 60.4 | 78.5 |
| DSpark mixed, total decode (tok/s) | 27.6 | 40.5 | 51.2 | 55.9 |

- `gufo bench` prefills 1,205 tok/s at depth 0 (synthetic token IDs).
- Repeating a source file with one edit decodes at 67.6 tok/s through
  prompt lookup.
- The server is ready 18.6 s after a cold start with the page cache dropped.
- From four users, plain AR is faster than DSpark on mixed text, because
  every session still drafts a full block (open work).
- Greedy DSpark and concurrent outputs are identical to single-session plain
  decoding.

## Status

Production target is Strix Halo `gfx1151` on Linux, like the rest of Gufo.
Next: streaming the `IQ2_XXS` experts from the SSD, so the stronger build
runs on the same machine. Development history and every measured experiment,
including rejected ones, are in
[EXPERIMENTS.md](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md) and the
[roadmap](docs/models/deepseek-v4.1-flash/ROADMAP.md).
