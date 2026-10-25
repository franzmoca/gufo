"""GGUF dequantization for the DeepSeek V4.1 Flash CPU oracle.

Covers every storage type in the ISTA GSQ-RCO builds and the Engram table:
F32, F16, BF16, Q8_0 and Q5_K come from ``tools/gufo/gguf.py``; this module adds
IQ1_S, IQ1_M, IQ2_XXS, IQ2_S and IQ4_NL. Codebooks are format constants, read
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
BLOCK.update({19: (256, 50), 29: (256, 56), 16: (256, 66), 22: (256, 82), 20: (32, 18)})


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


DEQUANT = dict(BASE_DEQUANT)
DEQUANT.update({19: dequant_iq1_s, 29: dequant_iq1_m, 16: dequant_iq2_xxs, 22: dequant_iq2_s, 20: dequant_iq4_nl})


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
