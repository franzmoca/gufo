# DeepSeek V4.1 Flash on one Strix Halo

Gufo runs **DeepSeek-V4.1-Flash-1017**, a 763B-parameter mixture-of-experts
model with a native 1M-token context, on a single AMD Strix Halo with 128 GB
of unified memory. The `IQ1_M` backbone stays resident in GPU memory; the
larger `IQ2_XXS` backbone can stream its routed experts from SSD. Engram
lookup tables are read from the SSD in both configurations.

> [!WARNING]
> **Headless, dedicated deployment only.** The resident model leaves about
> 1.75 GiB of host memory free once the server is ready (native context,
> one session, DSpark drafter). A graphical desktop, a browser or any other
> GPU workload on the same machine competes for that memory and can push
> the server or the system out of memory. Run it on a machine that does
> nothing else, without a desktop session, and reach it over the network.

## What it runs

- **Backbone:** ISTA-DASLab's GSQ-RCO `IQ1_M` build (1.75 bits per weight),
  resident in GTT, or the experimental `IQ2_XXS` build with streamed experts.
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
| Vision projector (`--mmproj`) | +0.90 GiB |
| Engram tables | 103 GiB on disk, not resident |
| **Host memory available** (1M context, one session, DSpark, server ready) | **1.75 GiB** |

The prompt cache sizes itself from what is left; in this configuration it
holds about 1.25 GB of checkpoints. The machine therefore has no room for
anything else: no desktop, no second model, no larger prefill batches
(`--prefill-batch 4096` leaves the prompt cache no memory at all).

For `IQ2_XXS`, Gufo keeps about 5.3 GiB of dense weights resident and uses a
memory-bounded expert cache. Cold prefill reads many experts from the SSD;
the cache, session state, drafter, vision tower and prompt cache share the
available memory. The engine reports cache hit rate, bytes read and host
wait in its logs.

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

All files go into **one directory**: Gufo opens the other shards from the
directory of the first one. The shared Engram file is shard 6 of every build,
through a link named after the build.

```sh
pip install -U "huggingface_hub[cli]"
mkdir -p ds41 && cd ds41

# Backbone (5 shards, 120.7 GB) and Engram tables (110.6 GB)
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ1_M*" "*Engram*" --local-dir .
ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00006-of-00006.gguf

# DSpark drafter (recommended: Q2_K/Q4_K, 4.9 GB)
hf download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF \
  DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf --local-dir .
```

| Source | File | Notes |
| --- | --- | --- |
| [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | `*-IQ1_M-0000{1..5}-of-00006.gguf` or `*-IQ2_XXS-0000{1..5}-of-00006.gguf`, `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (linked as shard 6) | `IQ1_M` fits resident; `IQ2_XXS` (138 GB) needs `--stream-experts`. |
| [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) | `DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf` | **Recommended drafter**: 4.59 GiB resident. |
| [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | Supported: 7.42 GiB resident, about 2.6 GiB less host memory left, similar speed and acceptance. |

Use the smaller Q2_K/Q4_K drafter: in matched runs the Q8_0 drafter
accepted as many drafts at similar speed, and the memory it takes comes out
of the prompt cache. Other published drafters (unsloth BF16, Lucebox
MXFP4/Q8) were not measured.

> [!WARNING]
> **Downloaded before October 27, 2026?** ISTA-DASLab republished the
> GSQ-RCO files in llama.cpp's layout (six shards per build, renamed metadata,
> the image routing bias included). Gufo loads only that layout and rejects
> the old four-shard files. Convert them in place, without downloading the
> model again, with the repository's
> [`convert_ds41_upstream.py`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/blob/main/convert_ds41_upstream.py)
> (Python 3.8+, no extra packages):
>
> ```sh
> python3 convert_ds41_upstream.py ds41 [--delete-old]
> ```
>
> It writes each build's new shards next to the old ones (121 to 155 GB of
> free space per build, or space for one build with `--delete-old`), rewrites
> only the 416-byte header of the Engram file and adds the shard-6 links.
> The quantized weights are copied byte for byte. A `mmproj-ds41-BF16.gguf`
> built with the former `tools/ds41/vision_sidecar.py` is no longer needed:
> pass the mmproj itself. Rename an expert histogram to the new first shard
> (`...-00001-of-00006.gguf.expert-histogram`).

## Vision tower

Image input needs the vision projector (0.97 GB); the backbone carries each
MoE gate's image routing bias:

```sh
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf --local-dir .
```

unsloth's `mmproj-BF16.gguf` works too. Vision is opt-in: the server loads it
only with an explicit `--mmproj` (0.90 GiB), never by discovery.

## Build and serve

Build as described in the [README](README.md#build-from-source), then:

```sh
./result/bin/gufo serve llm \
  -m ds41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf \
  --speculative dspark \
  --dspark-model ds41/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf \
  --host 0.0.0.0 --port 8080
```

Add `--mmproj ds41/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` for image input. The server is
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
  first token instead of 12 s). The first start with it hashes the weights and
  the Engram file (about 95 s); later starts reuse the hashes.

To try the larger quant, download its five shards beside the Engram file,
link the Engram file as its shard 6 and pass its first shard with
`--stream-experts`:

```sh
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ2_XXS*" --local-dir ds41
ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  ds41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf
./result/bin/gufo serve llm \
  -m ds41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  --stream-experts
```

Streaming is never chosen automatically: without `--stream-experts`, a model
that does not fit the available memory fails to load and states what it
needs. `--expert-cache-bytes N` bounds the streamed expert cache (default:
available memory minus 4 GiB); a smaller cache leaves more memory free and
reads more experts from the SSD.

`--stream-experts` also runs the `IQ1_M` build in less memory, more slowly.
`gufo bench` (pp2048, tg128, no drafter) and the host memory free once the
server is ready at a 256K context:

| `IQ1_M` | Free memory | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: | ---: |
| Resident | 5.6 GiB | 1,212 | 24.4 |
| Streamed, `--expert-cache-bytes 68719476736` (64 GiB) | 48.6 GiB | 752 | 17.9 |
| Streamed, 32 GiB cache | 78.8 GiB | 462 | 12.9 |
| Streamed, 16 GiB cache | 93.3 GiB | 384 | 9.7 |

Decode depends on how many routed experts the cache already holds, so it
varies with the text. The automatic prompt cache may later take up to half
of the free memory (at most 32 GiB); add `--cache-ram-bytes` to keep that
memory free as well.

Optionally seed the expert cache from recorded oracle routes. For example,
this histogram comes from the 3,000-token repository-documentation quality
test; representative routes from your own workload may be more useful:

```sh
nix develop -c python3 tools/ds41/expert_histogram.py \
  --model ds41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  docs/models/deepseek-v4.1-flash/artifacts/m3-streaming/iq2-oracle-routes3k.npz
```

The resulting sidecar is discovered automatically. Seeding reads experts
at startup; remove the sidecar to start with an empty expert cache.

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
  -m /models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf \
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

The following `IQ1_M` results are measured over HTTP with `gufo serve` (greedy, thinking off, real chat
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

## Model notes

V4.1 Flash against the V4 Flash engine it was forked from:

| Area | V4 Flash | V4.1 Flash |
|---|---|---|
| Shape | 43 layers, 4096 hidden, 256 experts, ffn 2048, q_lora 1024, 64 indexer heads | 40 layers (+3 DSpark layers not in the ISTA builds), 5120 hidden, 384 experts top-6 + 1 shared, ffn 2304, q_lora 1280, 32 indexer heads, RMS eps 1e-20 |
| Routing | 3 hash layers | none; sqrtsoftplus + bias top-6, normalized, ×1.5 |
| Layout | uniform | YoCo: layers 0-19 encoder, 20-39 decoder |
| Compression | ratio 4 (overlapping) / 128 | ratio 0 = SWA only (0-1); 2 = softmax-gated pool (2-19); 1 = RMSNorm(wkv(x)) (20-39). RoPE base 160000 + YaRN on ratio >= 1 |
| KV/index sharing | per layer | CSA2: layers 2, 8, 14, 20 own compressor and index K; 24, 28, 32, 36 rescore layer 20's index K with their own Q; all others reuse the latest top-512. Index K is projected from the main latent (512 -> 128) |
| Candidate blocks | none | layer 20 takes the block max over 8-row blocks, keeps the top 2048 blocks (newest pinned) and masks the rescoring layers; a no-op up to 16K context |
| mHC | mixes used on the same layer | shifted: attention uses the previous FFN's `pre`; layer 0 uses a one-hot `pre`; no `output_hc` |
| Engram | none | layers 1 and 14, before the block: 24 rows per token (2- to 4-grams × 8 heads over 99,092 compressed ids), `engram_kv` 6144 -> 5 × 5120, per-copy sigmoid gate, `h_c += gate_c * v` |
| Template | `<｜DSML｜tool_calls>` | DSML tags with a leading space, `Reasoning Effort: N` (1-100; high = 75 default), mid-conversation `<｜System｜>` turns |

State is small: about 890 B per token of compressed KV and index K plus a
fixed 128-row SWA ring per layer; stored as FP4 codes and scales it is about
0.9 GiB at 1M tokens. GGUF types: experts are IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S
per layer (gate and up share a type; down may differ); dense projections
Q8_0/Q5_K; router, compressor and indexer BF16; Engram gate (q·k), hc and
norms F32; `token_embd` and `output` Q5_K; Engram hash constants in metadata. An independent
CPU/torch oracle that ports the official `inference/model.py` and
`inference/engram.py` is in `tools/ds41/`.

## Status

Production target is Strix Halo `gfx1151` on Linux, like the rest of Gufo.
`IQ2_XXS` expert streaming is qualified for evaluation, with outputs
bit-identical to the staged build and the serving suites passing. How it works is in
[STREAMING.md](docs/models/deepseek-v4.1-flash/STREAMING.md); open work is in
the [backlog](docs/models/deepseek-v4.1-flash/BACKLOG.md), and every measured
experiment, including rejected ones, is in
[EXPERIMENTS.md](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md).
