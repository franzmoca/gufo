"""GGUF metadata and dequantization helpers for independent model oracles."""
from __future__ import annotations

import mmap
import numpy as np
from pathlib import Path
import struct

# --------------------------------------------------------------------------
# GGML value types (GGUF metadata)
# --------------------------------------------------------------------------
GGUF_TYPE = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64",
    12: "float64",
}

# --------------------------------------------------------------------------
# GGML tensor types (ggml.h enum). bpw + block codec per type.
# --------------------------------------------------------------------------
# bytes per name -> packing bytes
GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4",
}

# type -> (block_size, type_size_bytes)
GGML_TYPE_BLOCK = {
    0: (1, 4),     # f32
    1: (1, 2),     # f16
    2: (32, 18),   # q4_0: d + qs
    3: (32, 20),   # q4_1: d+m + qs
    6: (32, 22),   # q5_0
    7: (32, 24),   # q5_1
    8: (32, 34),   # q8_0
    10: (256, 2 + 2 + 16 + 64),  # q2_K
    11: (256, 2 + 64 + 32 + 12),  # q3_K
    12: (256, 4 + 12 + 128),   # q4_K  (d:2,dmin:2,scales:12,qs:128)
    13: (256, 4 + 12 + 32 + 128),  # q5_K
    14: (256, 2 + 128 + 64 + 16),  # q6_K
    15: (256, 4 + 256 + 32),    # q8_K
    16: (256, 2 + 64),   # iq2_xxs: d + qs[QK_K/8] as uint16
    20: (32, 18),   # iq4_nl
    21: (256, 2 + 64 + 8 + 32 + 4),  # iq3_s: d,qs,qh,signs,scales
    23: (256, 2 + 2 + 4 + 128),  # iq4_xs
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    30: (1, 2),    # bf16
    34: (256, 2 + 4 + (256 - 4 * 4) // 5),  # tq1_0
    35: (256, 2 + 64),   # tq2_0
}

# --------------------------------------------------------------------------
# Q4-family dequant codecs (llama.cpp ggml-quants.c, reference ports).
# QK_K = 256 super-block.
# --------------------------------------------------------------------------
QK_K = 256


def _fp16_to_fp32(u):
    # GGUF is written with native (host) endianness; x86 = little-endian.
    return np.frombuffer(u, dtype="<u2").astype(np.uint16).view("float16").astype(np.float32)


def _bf16_to_fp32(u):
    a = np.frombuffer(u, dtype="<u2").astype(np.uint32) << 16
    return a.view("float32")


def _dequant_f16(f):
    return _fp16_to_fp32(f)


def _dequant_bf16(f):
    return _bf16_to_fp32(f)


def _dequant_f32(f):
    return np.frombuffer(f, dtype="<f4").copy()


def _dequant_q4_0(f):
    nblk = len(f) // 18
    b = f.reshape(nblk, 18)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))  # (nblk,)
    qs = b[:, 2:].reshape(nblk, 16)
    lo = (qs & 0x0F).astype(np.float32) - 8
    hi = (qs >> 4).astype(np.float32) - 8
    out = np.empty((nblk, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.reshape(-1)


def _dequant_q8_0(f):
    nblk = len(f) // 34
    b = f.reshape(nblk, 34)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))
    qs = np.frombuffer(b[:, 2:].reshape(-1), dtype=np.int8).reshape(nblk, 32).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


def _dequant_q4_K(f):
    # block_q4_K: d(2) dmin(2) scales(12) qs(128)
    nblk = len(f) // 144
    b = f.reshape(nblk, 144)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))
    dmin = _fp16_to_fp32(b[:, 2:4].reshape(-1))
    scales = b[:, 4:16]              # (nblk,12)
    qs = b[:, 16:144]                # (nblk,128)
    # 8 per-32-block (d,m) 6-bit pairs
    d8 = np.empty((nblk, 8), dtype=np.int32)
    m8 = np.empty((nblk, 8), dtype=np.int32)
    for j in range(4):
        d8[:, j] = scales[:, j] & 63
        m8[:, j] = scales[:, j + 4] & 63
    for j in range(4, 8):
        d8[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m8[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    out = np.empty((nblk, QK_K), dtype=np.float32)
    for k in range(4):
        d1 = d * d8[:, 2 * k].astype(np.float32); m1 = dmin * m8[:, 2 * k].astype(np.float32)
        d2 = d * d8[:, 2 * k + 1].astype(np.float32); m2 = dmin * m8[:, 2 * k + 1].astype(np.float32)
        q = qs[:, k * 32:(k + 1) * 32]
        lo = (q & 0x0F).astype(np.float32)
        hi = (q >> 4).astype(np.float32)
        out[:, k * 64:k * 64 + 32] = lo * d1[:, None] - m1[:, None]
        out[:, k * 64 + 32:k * 64 + 64] = hi * d2[:, None] - m2[:, None]
    return out.reshape(-1)


def _dequant_q5_K(f):
    # block_q5_K: d(2) dmin(2) scales(12) qh(32) qs(128)
    nblk = len(f) // 176
    b = f.reshape(nblk, 176)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))
    dmin = _fp16_to_fp32(b[:, 2:4].reshape(-1))
    scales = b[:, 4:16]
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    d8 = np.empty((nblk, 8), dtype=np.int32)
    m8 = np.empty((nblk, 8), dtype=np.int32)
    for j in range(4):
        d8[:, j] = scales[:, j] & 63
        m8[:, j] = scales[:, j + 4] & 63
    for j in range(4, 8):
        d8[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m8[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    out = np.empty((nblk, QK_K), dtype=np.float32)
    for k in range(4):
        d1 = d * d8[:, 2 * k].astype(np.float32); m1 = dmin * m8[:, 2 * k].astype(np.float32)
        d2 = d * d8[:, 2 * k + 1].astype(np.float32); m2 = dmin * m8[:, 2 * k + 1].astype(np.float32)
        q = qs[:, k * 32:(k + 1) * 32]
        h = qh[:, 0:32]
        b1 = ((h & (1 << (2 * k))) != 0).astype(np.float32) * 16
        b2 = ((h & (2 << (2 * k))) != 0).astype(np.float32) * 16
        lo = (q & 0x0F).astype(np.float32) + b1
        hi = (q >> 4).astype(np.float32) + b2
        out[:, k * 64:k * 64 + 32] = lo * d1[:, None] - m1[:, None]
        out[:, k * 64 + 32:k * 64 + 64] = hi * d2[:, None] - m2[:, None]
    return out.reshape(-1)


def _dequant_q6_K(f):
    # block_q6_K: ql(128) qh(64) scales(16) d(2)
    nblk = len(f) // 210
    b = f.reshape(nblk, 210)
    d = _fp16_to_fp32(b[:, 208:210].reshape(-1))
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    sc = np.frombuffer(b[:, 192:208].reshape(-1), dtype=np.int8).reshape(nblk, 16).astype(np.int32)
    out = np.empty((nblk, QK_K), dtype=np.float32)
    for g in range(2):  # two 128-lane halves
        qlg = ql[:, g * 64:(g + 1) * 64]
        qhg = qh[:, g * 32:(g + 1) * 32]
        scg = sc[:, g * 8:(g + 1) * 8]
        sel = (np.arange(32) // 16)
        q1 = ((qlg[:, 0:32] & 0x0F) | ((qhg & 0x03) << 4)).astype(np.int32) - 32
        q2 = ((qlg[:, 32:64] & 0x0F) | ((qhg >> 2 & 0x03) << 4)).astype(np.int32) - 32
        q3 = ((qlg[:, 0:32] >> 4) | ((qhg >> 4 & 0x03) << 4)).astype(np.int32) - 32
        q4 = ((qlg[:, 32:64] >> 4) | ((qhg >> 6 & 0x03) << 4)).astype(np.int32) - 32
        s1 = scg[:, sel + 0].astype(np.float32)
        s2 = scg[:, sel + 2].astype(np.float32)
        s3 = scg[:, sel + 4].astype(np.float32)
        s4 = scg[:, sel + 6].astype(np.float32)
        base = g * 128
        out[:, base + 0:base + 32] = d[:, None] * s1 * q1
        out[:, base + 32:base + 64] = d[:, None] * s2 * q2
        out[:, base + 64:base + 96] = d[:, None] * s3 * q3
        out[:, base + 96:base + 128] = d[:, None] * s4 * q4
    return out.reshape(-1)


DEQUANT = {
    0: _dequant_f32,
    1: _dequant_f16,
    2: _dequant_q4_0,
    8: _dequant_q8_0,
    12: _dequant_q4_K,
    13: _dequant_q5_K,
    14: _dequant_q6_K,
    30: _dequant_bf16,
}


def bpw(t):
    bs, ts = GGML_TYPE_BLOCK[t]
    return ts * 8 / bs


# --------------------------------------------------------------------------
# GGUF binary reader
# --------------------------------------------------------------------------
class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def u8(self): v = self.data[self.pos]; self.pos += 1; return v
    def i8(self): return struct.unpack_from("<b", self.data, self.pos)[0] if self._take(1) else 0
    def u16(self): v = struct.unpack_from("<H", self.data, self.pos)[0]; self.pos += 2; return v
    def i16(self): v = struct.unpack_from("<h", self.data, self.pos)[0]; self.pos += 2; return v
    def u32(self): v = struct.unpack_from("<I", self.data, self.pos)[0]; self.pos += 4; return v
    def i32(self): v = struct.unpack_from("<i", self.data, self.pos)[0]; self.pos += 4; return v
    def u64(self): v = struct.unpack_from("<Q", self.data, self.pos)[0]; self.pos += 8; return v
    def i64(self): v = struct.unpack_from("<q", self.data, self.pos)[0]; self.pos += 8; return v
    def f32(self): v = struct.unpack_from("<f", self.data, self.pos)[0]; self.pos += 4; return v
    def f64(self): v = struct.unpack_from("<d", self.data, self.pos)[0]; self.pos += 8; return v

    def _take(self, n):
        assert self.pos + n <= len(self.data), f"overrun @{self.pos}+{n}"
        return self.pos + n <= len(self.data)

    def string(self):
        ln = self.u64()
        s = self.data[self.pos:self.pos + ln]
        self.pos += ln
        return s.decode("utf-8", "replace")

    def read_value(self, tag):
        if tag == 0: return self.u8()
        if tag == 1: return self.i8()
        if tag == 2: return self.u16()
        if tag == 3: return self.i16()
        if tag == 4: return self.u32()
        if tag == 5: return self.i32()
        if tag == 6: return self.f32()
        if tag == 7: return bool(self.u8())
        if tag == 8: return self.string()
        if tag == 9:
            atype = self.u32()
            nelem = self.u64()
            return [self.read_value(atype) for _ in range(nelem)]
        if tag == 10: return self.u64()
        if tag == 11: return self.i64()
        if tag == 12: return self.f64()
        raise ValueError(f"unknown gguf value tag {tag}")

    def skip_value(self, tag):
        if tag == 9:
            atype = self.u32()
            nelem = self.u64()
            for _ in range(nelem):
                self.skip_value(atype)
            return
        if tag == 8:
            ln = self.u64()
            self.pos += ln
            return
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        self.pos += sizes[tag]
        if self.pos > len(self.data):
            raise ValueError("value overruns buffer")


def parse_gguf(path):
    with Path(path).open("rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return _parse_gguf_data(data)


def _parse_gguf_data(data):
    assert data[0:4] == b"GGUF", "not a GGUF file"
    r = Reader(data)
    r.pos = 4
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    metadata = {}
    user_kv = {}
    for _ in range(n_kv):
        key = r.string()
        tag = r.u32()
        if key.startswith("tokenizer."):
            # vocab arrays are huge and unneeded for inspection; skip without storing.
            r.skip_value(tag)
            continue
        val = r.read_value(tag)
        if r.pos > 262144:
            pass  # ok, bigger metadata is fine if the file is fully read
        if key.startswith("general."):
            metadata[key] = val
        else:
            user_kv[key] = val

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        # dims are packed as u64 (ggml ne[] are int64); file order = innermost first.
        ne = tuple(r.u64() for _ in range(n_dims))
        ttype = r.u32()
        offset = r.u64()
        tensors.append({
            "name": name, "shape": list(reversed(ne)), "type": ttype,
            "type_name": GGML_TYPE_NAME.get(ttype, f"T{ttype}"),
            "offset": offset,
        })

    # data section is aligned per general.alignment (writer pads after TI data)
    alignment = metadata.get("general.alignment", 32)
    data_offset = (r.pos + alignment - 1) // alignment * alignment
    return {
        "version": ver, "n_tensors": n_tensors, "metadata": metadata,
        "user_kv": user_kv, "tensors": tensors, "data_offset": data_offset,
        "file_size": len(data),
    }


def tensor_bytes(state, t):
    bs, ts = GGML_TYPE_BLOCK.get(t["type"], (None, None))
    if bs is None:
        return None
    ne = 1
    for d in t["shape"]:
        ne *= d
    nblk = ne // bs
    return nblk * ts


def write_gguf(path, metadata, tensors, alignment=32):
    """Writes a GGUF v3 file of F32 tensors.

    metadata: (key, value) pairs; str, bool, int (uint32) and float (float32).
    tensors: (name, array) pairs; arrays are stored as F32 with the
    outermost dimension first (numpy order), so ne is the reversed shape.
    """
    def string(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def value(v):
        if isinstance(v, bool):
            return struct.pack("<IB", 7, v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + string(v)
        if isinstance(v, int):
            return struct.pack("<II", 4, v)
        if isinstance(v, float):
            return struct.pack("<If", 6, v)
        raise TypeError(f"unsupported metadata value {v!r}")

    arrays = [(name, np.ascontiguousarray(a, dtype="<f4")) for name, a in tensors]
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(arrays), len(metadata)))
    for key, v in metadata:
        head += string(key) + value(v)
    offset = 0
    for name, a in arrays:
        head += string(name) + struct.pack("<I", a.ndim)
        head += b"".join(struct.pack("<Q", d) for d in reversed(a.shape))
        head += struct.pack("<IQ", 0, offset)
        offset += (a.nbytes + alignment - 1) // alignment * alignment
    head += bytes(-len(head) % alignment)
    with Path(path).open("wb") as out:
        out.write(head)
        for _, a in arrays:
            out.write(a.tobytes())
            out.write(bytes(-a.nbytes % alignment))
