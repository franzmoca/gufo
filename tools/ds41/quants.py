"""GGUF dequantization for the DeepSeek V4.1 Flash CPU oracle.

Covers every storage type in the ISTA GSQ-RCO builds and the Engram table:
F32, F16, BF16, Q8_0 and Q5_K come from ``tools/gufo/gguf.py``; this module adds
IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S and IQ4_NL, and MXFP4 for the DSpark drafter's experts. Codebooks are format constants, read
from the vendored ggml header so the oracle and the kernels agree on the format
while sharing no decode code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo.gguf import DEQUANT as BASE_DEQUANT  # noqa: E402
from gufo.gguf import GGML_TYPE_BLOCK as BASE_BLOCK  # noqa: E402

GGML_COMMON = ROOT / "src/models/deepseek_v4_flash/kernels/rocm/mmq/ggml-common.h"
QK_K = 256
IQ1_DELTA = 0.125

# type -> (elements per block, bytes per block)
BLOCK = dict(BASE_BLOCK)
BLOCK.update({19: (256, 50), 29: (256, 56), 16: (256, 66), 17: (256, 74), 22: (256, 82),
              20: (32, 18), 39: (32, 17)})


@lru_cache(None)
def _table(name: str) -> np.ndarray:
    text = GGML_COMMON.read_text()
    match = re.search(r"GGML_TABLE_BEGIN\((\w+), " + name + r", [^)]*\)(.*?)GGML_TABLE_END", text, re.S)
    if match is None:
        raise KeyError(name)
    ctype, body = match.groups()
    values = [int(v, 0) for v in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", body)]
    dtype = {"uint64_t": np.uint64, "uint32_t": np.uint32, "uint8_t": np.uint8, "int8_t": np.int8}[ctype]
    return np.array(values, dtype=np.int64).astype(dtype)


@lru_cache(None)
def _iq1s_grid() -> np.ndarray:
    """2048 x 8 values in {-1, 0, 1}. The GPU table stores value+1 in nibbles: elements
    0..3 in the low nibble of bytes 0..3, elements 4..7 in the high nibble."""
    packed = _table("iq1s_grid_gpu").astype(np.uint32)
    shifts = np.array([0, 8, 16, 24, 4, 12, 20, 28], dtype=np.uint32)
    return ((packed[:, None] >> shifts) & 0xF).astype(np.float32) - 1.0


@lru_cache(None)
def _byte_grid(name: str) -> np.ndarray:
    packed = _table(name).astype(np.uint64)
    shifts = np.arange(8, dtype=np.uint64) * np.uint64(8)
    return ((packed[:, None] >> shifts) & np.uint64(0xFF)).astype(np.float32)


@lru_cache(None)
def _sign_table() -> np.ndarray:
    """128 sign patterns x 8 multipliers (+1 / -1)."""
    signs = _table("ksigns_iq2xs").astype(np.uint8)
    bits = (signs[:, None] >> np.arange(8, dtype=np.uint8)) & 1
    return 1.0 - 2.0 * bits.astype(np.float32)


def _f16(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<f2").astype(np.float32).reshape(raw.shape[:-1])


def _blocks(data: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(data, dtype=np.uint8).reshape(-1, size)


def dequant_iq1_s(data: np.ndarray) -> np.ndarray:
    b = _blocks(data, 50)
    d = _f16(b[:, 0:2])
    qs = b[:, 2:34].reshape(-1, 8, 4).astype(np.int64)
    qh = np.ascontiguousarray(b[:, 34:50]).view("<u2").astype(np.int64)  # [nb, 8]
    shift = 3 * np.arange(4)
    index = qs | (((qh[:, :, None] >> shift) & 7) << 8)
    scale = d[:, None] * (2 * ((qh >> 12) & 7) + 1)
    delta = np.where(qh & 0x8000, -IQ1_DELTA, IQ1_DELTA)
    values = _iq1s_grid()[index] + delta[:, :, None, None]
    return (values * scale[:, :, None, None]).reshape(-1).astype(np.float32)


def dequant_iq1_m(data: np.ndarray) -> np.ndarray:
    b = _blocks(data, 56)
    qs = b[:, 0:32].reshape(-1, 8, 4).astype(np.int64)
    qh = b[:, 32:48].reshape(-1, 8, 2).astype(np.int64)
    sc = np.ascontiguousarray(b[:, 48:56]).view("<u2").astype(np.int64)  # [nb, 4]
    packed = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000)
    d = packed.astype(np.uint16).view("<f2").astype(np.float32)
    ib = np.arange(8)
    word = sc[:, ib // 2] >> (6 * (ib % 2))  # [nb, 8]
    scale1 = d[:, None] * (2 * (word & 7) + 1)
    scale2 = d[:, None] * (2 * ((word >> 3) & 7) + 1)
    high = np.stack([(qh[..., 0] << 8), (qh[..., 0] << 4), (qh[..., 1] << 8), (qh[..., 1] << 4)], axis=-1) & 0x700
    index = qs | high
    flag = np.stack([qh[..., 0] & 0x08, qh[..., 0] & 0x80, qh[..., 1] & 0x08, qh[..., 1] & 0x80], axis=-1)
    delta = np.where(flag != 0, -IQ1_DELTA, IQ1_DELTA)
    values = _iq1s_grid()[index] + delta[..., None]
    scale = np.stack([scale1, scale1, scale2, scale2], axis=-1)
    return (values * scale[..., None]).reshape(-1).astype(np.float32)


def dequant_iq2_xxs(data: np.ndarray) -> np.ndarray:
    b = _blocks(data, 66)
    d = _f16(b[:, 0:2])
    groups = b[:, 2:66].reshape(-1, 8, 8)
    index = groups[:, :, 0:4].astype(np.int64)
    aux = np.ascontiguousarray(groups[:, :, 4:8]).view("<u4").reshape(-1, 8).astype(np.int64)
    scale = d[:, None] * (0.5 + (aux >> 28)) * 0.25
    sign_index = (aux[:, :, None] >> (7 * np.arange(4))) & 127
    values = _byte_grid("iq2xxs_grid")[index] * _sign_table()[sign_index]
    return (values * scale[:, :, None, None]).reshape(-1).astype(np.float32)


def dequant_iq2_xs(data: np.ndarray) -> np.ndarray:
    b = _blocks(data, 74)
    d = _f16(b[:, 0:2])
    qs = np.ascontiguousarray(b[:, 2:66]).view("<u2").reshape(-1, 8, 4).astype(np.int64)
    scales = b[:, 66:74].astype(np.int64)
    low = d[:, None] * (0.5 + (scales & 0xF)) * 0.25
    high = d[:, None] * (0.5 + (scales >> 4)) * 0.25
    scale = np.stack([low, low, high, high], axis=-1)
    values = _byte_grid("iq2xs_grid")[qs & 0x1FF] * _sign_table()[qs >> 9]
    return (values * scale[..., None]).reshape(-1).astype(np.float32)


def dequant_iq2_s(data: np.ndarray) -> np.ndarray:
    b = _blocks(data, 82)
    d = _f16(b[:, 0:2])
    qs = b[:, 2:34].reshape(-1, 8, 4).astype(np.int64)
    signs = b[:, 34:66].reshape(-1, 8, 4).astype(np.uint8)
    qh = b[:, 66:74].astype(np.int64)
    scales = b[:, 74:82].astype(np.int64)
    index = qs | ((qh[:, :, None] << (8 - 2 * np.arange(4))) & 0x300)
    low = d[:, None] * (0.5 + (scales & 0xF)) * 0.25
    high = d[:, None] * (0.5 + (scales >> 4)) * 0.25
    scale = np.stack([low, low, high, high], axis=-1)
    bits = (signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    values = _byte_grid("iq2s_grid")[index] * (1.0 - 2.0 * bits.astype(np.float32))
    return (values * scale[..., None]).reshape(-1).astype(np.float32)


def dequant_iq4_nl(data: np.ndarray) -> np.ndarray:
    b = _blocks(data, 18)
    d = _f16(b[:, 0:2])
    qs = b[:, 2:18]
    codebook = _table("kvalues_iq4nl").astype(np.float32)
    values = np.concatenate([codebook[qs & 0xF], codebook[qs >> 4]], axis=-1)
    return (values * d[:, None]).reshape(-1).astype(np.float32)


def dequant_mxfp4(data: np.ndarray) -> np.ndarray:
    """E8M0 exponent e, then 32 E2M1 codes (low nibbles first); the table holds doubled values, so
    the scale is 2**(e - 128)."""
    b = _blocks(data, 17)
    scale = np.ldexp(1.0, b[:, 0].astype(np.int64) - 128)
    qs = b[:, 1:17]
    codebook = _table("kvalues_mxfp4").astype(np.float64)
    values = np.concatenate([codebook[qs & 0xF], codebook[qs >> 4]], axis=-1)
    return (values * scale[:, None]).reshape(-1).astype(np.float32)


# Torch ports of the expert formats: the same integer decode and the same float64
# scale product rounded once to float32, so they match the numpy versions bit for
# bit while running on the oracle's device.
def _t(table: np.ndarray, device) -> "torch.Tensor":
    import torch

    return torch.from_numpy(np.ascontiguousarray(table)).to(device)


def _t_f16(raw):
    import torch

    return raw.contiguous().view(torch.float16).double().reshape(raw.shape[:-1])


def _t_u16(raw):
    import torch

    return raw.contiguous().view(torch.int16).long() & 0xFFFF


def _t_u32(raw):
    import torch

    return raw.contiguous().view(torch.int32).long() & 0xFFFFFFFF


def torch_dequant_iq1_s(b):
    import torch

    b = b.reshape(-1, 50)
    d = _t_f16(b[:, 0:2])
    qs = b[:, 2:34].reshape(-1, 8, 4).long()
    qh = _t_u16(b[:, 34:50])
    shift = 3 * torch.arange(4, device=b.device)
    index = qs | (((qh[:, :, None] >> shift) & 7) << 8)
    scale = d[:, None] * (2 * ((qh >> 12) & 7) + 1)
    delta = torch.where((qh & 0x8000) != 0, -IQ1_DELTA, IQ1_DELTA).float()
    values = _t(_iq1s_grid(), b.device)[index] + delta[:, :, None, None]
    return (values.double() * scale[:, :, None, None]).reshape(-1).float()


def torch_dequant_iq1_m(b):
    import torch

    b = b.reshape(-1, 56)
    qs = b[:, 0:32].reshape(-1, 8, 4).long()
    qh = b[:, 32:48].reshape(-1, 8, 2).long()
    sc = _t_u16(b[:, 48:56])
    packed = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000)
    d = packed.to(torch.int16).view(torch.float16).double()
    ib = torch.arange(8, device=b.device)
    word = sc[:, ib // 2] >> (6 * (ib % 2))
    scale1 = d[:, None] * (2 * (word & 7) + 1)
    scale2 = d[:, None] * (2 * ((word >> 3) & 7) + 1)
    high = torch.stack([(qh[..., 0] << 8), (qh[..., 0] << 4), (qh[..., 1] << 8), (qh[..., 1] << 4)], -1) & 0x700
    index = qs | high
    flag = torch.stack([qh[..., 0] & 0x08, qh[..., 0] & 0x80, qh[..., 1] & 0x08, qh[..., 1] & 0x80], -1)
    delta = torch.where(flag != 0, -IQ1_DELTA, IQ1_DELTA).float()
    values = _t(_iq1s_grid(), b.device)[index] + delta[..., None]
    scale = torch.stack([scale1, scale1, scale2, scale2], -1)
    return (values.double() * scale[..., None]).reshape(-1).float()


def torch_dequant_iq2_xxs(b):
    import torch

    b = b.reshape(-1, 66)
    d = _t_f16(b[:, 0:2])
    groups = b[:, 2:66].reshape(-1, 8, 8)
    index = groups[:, :, 0:4].long()
    aux = _t_u32(groups[:, :, 4:8]).reshape(-1, 8)
    scale = d[:, None] * (0.5 + (aux >> 28)) * 0.25
    sign_index = (aux[:, :, None] >> (7 * torch.arange(4, device=b.device))) & 127
    values = _t(_byte_grid("iq2xxs_grid"), b.device)[index] * _t(_sign_table(), b.device)[sign_index]
    return (values.double() * scale[:, :, None, None]).reshape(-1).float()


def torch_dequant_iq2_xs(b):
    import torch

    b = b.reshape(-1, 74)
    d = _t_f16(b[:, 0:2])
    q = b[:, 2:66].reshape(-1, 8, 4, 2).long()
    qs = q[..., 0] | (q[..., 1] << 8)
    scales = b[:, 66:74].long()
    low = d[:, None] * (0.5 + (scales & 0xF)) * 0.25
    high = d[:, None] * (0.5 + (scales >> 4)) * 0.25
    scale = torch.stack([low, low, high, high], -1)
    values = _t(_byte_grid("iq2xs_grid"), b.device)[qs & 0x1FF] * _t(_sign_table(), b.device)[qs >> 9]
    return (values.double() * scale[..., None]).reshape(-1).float()


def torch_dequant_iq2_s(b):
    import torch

    b = b.reshape(-1, 82)
    d = _t_f16(b[:, 0:2])
    qs = b[:, 2:34].reshape(-1, 8, 4).long()
    signs = b[:, 34:66].reshape(-1, 8, 4)
    qh = b[:, 66:74].long()
    scales = b[:, 74:82].long()
    index = qs | ((qh[:, :, None] << (8 - 2 * torch.arange(4, device=b.device))) & 0x300)
    low = d[:, None] * (0.5 + (scales & 0xF)) * 0.25
    high = d[:, None] * (0.5 + (scales >> 4)) * 0.25
    scale = torch.stack([low, low, high, high], -1)
    bits = (signs[..., None] >> torch.arange(8, dtype=torch.uint8, device=b.device)) & 1
    values = _t(_byte_grid("iq2s_grid"), b.device)[index] * (1.0 - 2.0 * bits.float())
    return (values.double() * scale[..., None]).reshape(-1).float()


TORCH_DEQUANT = {19: torch_dequant_iq1_s, 29: torch_dequant_iq1_m, 16: torch_dequant_iq2_xxs,
                 17: torch_dequant_iq2_xs, 22: torch_dequant_iq2_s}

DEQUANT = dict(BASE_DEQUANT)
DEQUANT.update({19: dequant_iq1_s, 29: dequant_iq1_m, 16: dequant_iq2_xxs, 17: dequant_iq2_xs,
                22: dequant_iq2_s, 20: dequant_iq4_nl, 39: dequant_mxfp4})


def nbytes(ggml_type: int, elements: int) -> int:
    per_block, size = BLOCK[ggml_type]
    if elements % per_block:
        raise ValueError(f"{elements} elements are not a whole number of type-{ggml_type} blocks")
    return elements // per_block * size


def dequantize(ggml_type: int, data: np.ndarray, elements: int) -> np.ndarray:
    if ggml_type == 26:
        return np.frombuffer(data, dtype="<i4", count=elements).copy()
    if ggml_type == 27:
        return np.frombuffer(data, dtype="<i8", count=elements).copy()
    values = DEQUANT[ggml_type](np.frombuffer(data, dtype=np.uint8, count=nbytes(ggml_type, elements)))
    return values[:elements]
