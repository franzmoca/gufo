"""Memory-mapped access to a split DeepSeek V4.1 GGUF and its separate Engram file."""
from __future__ import annotations

import mmap
from pathlib import Path
import re

import numpy as np

from quants import dequantize, nbytes
from gufo.gguf import Reader, parse_gguf


class TensorFile:
    def __init__(self, path: Path):
        self.path = path
        self.info = parse_gguf(path)
        self._file = path.open("rb")
        self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.kv = {**self.info["metadata"], **self.info["user_kv"]}


class Tensor:
    def __init__(self, source: TensorFile, record: dict):
        self.source = source
        self.name = record["name"]
        self.shape = tuple(record["shape"])  # outermost first (torch order)
        self.type = record["type"]
        self.type_name = record["type_name"]
        self.start = source.info["data_offset"] + record["offset"]

    @property
    def elements(self) -> int:
        return int(np.prod(self.shape))

    def raw(self, first_row: int = 0, rows: int | None = None) -> memoryview:
        """Bytes of rows [first_row, first_row + rows) along the outermost dimension."""
        row_elements = self.elements // self.shape[0]
        row_bytes = nbytes(self.type, row_elements)
        rows = self.shape[0] - first_row if rows is None else rows
        start = self.start + first_row * row_bytes
        return memoryview(self.source.data)[start:start + rows * row_bytes]

    def array(self, first_row: int = 0, rows: int | None = None) -> np.ndarray:
        rows = self.shape[0] - first_row if rows is None else rows
        row_elements = self.elements // self.shape[0]
        values = dequantize(self.type, self.raw(first_row, rows), rows * row_elements)
        return values.reshape((rows, *self.shape[1:]))


class Model:
    """All shards of one build plus the Engram tables named by ``deepseek41.engram.file``."""

    def __init__(self, first_shard: Path, engram: Path | None = None):
        first_shard = Path(first_shard)
        match = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first_shard.name)
        paths = [first_shard]
        if match:
            count = int(match.group(2))
            paths = [first_shard.with_name(f"{match.group(1)}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
        self.files = [TensorFile(p) for p in paths]
        self.kv = self.files[0].kv
        if self.kv.get("general.architecture") != "deepseek41":
            raise ValueError(f"{first_shard}: architecture {self.kv.get('general.architecture')!r} is not deepseek41")
        self.tensors: dict[str, Tensor] = {}
        for source in self.files:
            for record in source.info["tensors"]:
                self.tensors[record["name"]] = Tensor(source, record)
        expected = self.kv.get("split.tensors.count")
        if expected is not None and expected != len(self.tensors):
            raise ValueError(f"expected {expected} tensors across shards, found {len(self.tensors)}")
        if self.kv.get("deepseek41.engram.present"):
            engram = Path(engram) if engram else first_shard.with_name(self.kv["deepseek41.engram.file"])
            table = TensorFile(engram)
            if table.kv.get("deepseek41.engram.source") != self.kv.get("deepseek41.engram.source"):
                raise ValueError(f"{engram}: Engram source does not match the backbone")
            for record in table.info["tensors"]:
                self.tensors[record["name"]] = Tensor(table, record)

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def metadata_value(self, key: str):
        """Read one metadata value, including the tokenizer arrays ``parse_gguf`` skips."""
        r = Reader(self.files[0].data)
        r.pos = 8
        r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            name, tag = r.string(), r.u32()
            if name == key:
                return r.read_value(tag)
            r.skip_value(tag)
        raise KeyError(key)

    def meta(self, key: str, default=None):
        return self.kv.get("deepseek41." + key, default)
