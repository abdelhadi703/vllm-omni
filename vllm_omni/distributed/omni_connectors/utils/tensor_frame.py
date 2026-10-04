# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owned CPU tensor trees with a msgpack header and raw tensor regions.

The format uses tagged nodes rather than reserved dictionary keys. It never
exposes a view of transport storage to a caller. Unsupported objects retain
the ordinary OmniSerializer path.
"""

from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass

import msgspec
import torch

_LENGTH = struct.Struct("<I")
_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder()
_SCALAR, _DICT, _LIST, _TENSOR = range(4)


def _align(size: int) -> int:
    return (size + 15) & ~15


class _UnsupportedTreeError(Exception):
    pass


@dataclass
class TensorFrame:
    header: bytes
    tensors: list[torch.Tensor]
    offsets: list[int]
    size: int

    def write(self, buffer: memoryview, start: int) -> None:
        """Copy directly into caller-owned writable storage, while it is locked."""
        _LENGTH.pack_into(buffer, start, len(self.header))
        buffer[start + 4 : start + 4 + len(self.header)] = self.header
        data_start = start + _align(4 + len(self.header))
        # c_char is an anchor only; all bounds are checked by the ring writer.
        anchor = ctypes.c_char.from_buffer(buffer)
        base = ctypes.addressof(anchor) + data_start
        for tensor, offset in zip(self.tensors, self.offsets, strict=True):
            if tensor.nbytes:
                ctypes.memmove(base + offset, tensor.data_ptr(), tensor.nbytes)


def prepare_tensor_frame(payload: object, *, max_bytes: int | None = None) -> TensorFrame | None:
    """Prepare a plain tree containing dense CPU tensors; otherwise return None.

    Retaining tensors is enough because the caller writes before put() returns.
    This is not an asynchronous snapshot or a CUDA source ownership protocol.
    """
    tensors: list[torch.Tensor] = []
    offsets: list[int] = []
    descriptors: list[tuple[str, tuple[int, ...], int]] = []
    data_size = 0

    def visit(value: object, depth: int) -> list:
        nonlocal data_size
        if depth > 64:
            raise _UnsupportedTreeError
        if type(value) in (type(None), bool, int, float, str, bytes):
            return [_SCALAR, value]
        if type(value) is torch.Tensor:
            if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
                raise _UnsupportedTreeError
            tensor = value
            offset = _align(data_size)
            index = len(tensors)
            tensors.append(tensor)
            offsets.append(offset)
            descriptors.append((str(tensor.dtype).removeprefix("torch."), tuple(tensor.shape), offset))
            data_size = offset + tensor.nbytes
            if max_bytes is not None and data_size > max_bytes:
                raise _UnsupportedTreeError
            return [_TENSOR, index]
        if type(value) is dict and all(type(key) is str for key in value):
            return [_DICT, {key: visit(item, depth + 1) for key, item in value.items()}]
        if isinstance(value, (list, tuple)) and type(value) in (list, tuple):
            # msgpack represents both as arrays; match OmniSerializer's wire type.
            return [_LIST, [visit(item, depth + 1) for item in value]]
        if isinstance(value, msgspec.Struct):
            try:
                fields = msgspec.to_builtins(value, builtin_types=(torch.Tensor, bytes))
            except TypeError:
                raise _UnsupportedTreeError from None
            return visit(fields, depth + 1)
        raise _UnsupportedTreeError

    try:
        tree = visit(payload, 0)
        if not tensors:
            return None
        header = _ENCODER.encode([descriptors, tree])
    except (_UnsupportedTreeError, OverflowError):
        return None
    size = _align(4 + len(header)) + data_size
    if max_bytes is not None and size > max_bytes:
        return None
    # Only materialize non-contiguous/flagged storage after the entire tree
    # is known to fit. Large and unsupported payloads avoid an unused copy.
    for index, tensor in enumerate(tensors):
        if tensor.is_conj():
            tensor = tensor.resolve_conj()
        if tensor.is_neg():
            tensor = tensor.resolve_neg()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensors[index] = tensor
    return TensorFrame(header, tensors, offsets, size)


def read_tensor_frame(buffer: memoryview) -> object:
    """Decode into independent CPU allocations before returning ring credit."""
    if len(buffer) < 4:
        raise ValueError("truncated tensor frame")
    header_size = _LENGTH.unpack_from(buffer)[0]
    data_start = _align(4 + header_size)
    if data_start > len(buffer):
        raise ValueError("tensor frame header exceeds payload")
    header_view = buffer[4 : 4 + header_size]
    try:
        descriptors, tree = _DECODER.decode(header_view)
    finally:
        header_view.release()
    # All leaves share one independently owned allocation. This amortizes CPU
    # allocation/copy dispatch across small codec groups and hidden-state leaves.
    data_size = len(buffer) - data_start
    # A small metadata tensor must not keep a large, already consumed hidden
    # state alive. Pack small trees; give large multi-leaf trees separate owners.
    storage = torch.empty(data_size, dtype=torch.uint8) if data_size <= 65536 or len(descriptors) == 1 else None
    anchor = ctypes.c_char.from_buffer(buffer)
    if data_size and storage is not None:
        ctypes.memmove(storage.data_ptr(), ctypes.addressof(anchor) + data_start, data_size)
    tensors = []
    for name, shape, offset in descriptors:
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("unknown tensor frame dtype")
        if any(type(dim) is not int or dim < 0 for dim in shape) or type(offset) is not int or offset < 0:
            raise ValueError("invalid tensor frame region")
        count = math.prod(shape)
        size = count * dtype.itemsize
        start = data_start + offset
        if start + size > len(buffer):
            raise ValueError("tensor frame region exceeds payload")
        if size and storage is not None:
            tensor = storage[offset : offset + size].view(dtype).reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
            if size:
                ctypes.memmove(tensor.data_ptr(), ctypes.addressof(anchor) + start, size)
        tensors.append(tensor)

    def restore(node: list) -> object:
        tag, value = node
        if tag == _SCALAR:
            return value
        if tag == _TENSOR:
            return tensors[value]
        if tag == _DICT:
            return {key: restore(item) for key, item in value.items()}
        if tag == _LIST:
            return [restore(item) for item in value]
        raise ValueError("unknown tensor frame node")

    return restore(tree)
