# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeVar

from ._bytes import Byteorder, TypeTag
from ._files import open_binary
from ._low_level_io import LowLevelSqw
from ._models import (
    SqwDataBlockDescriptor,
    SqwDataBlockType,
    SqwFileHeader,
    SqwFileType,
)

_T = TypeVar("_T")


class SQW:
    def __init__(
        self,
        *,
        sqw_io: LowLevelSqw,
        file_header: SqwFileHeader,
        data_block_descriptors: dict[tuple[str, str], SqwDataBlockDescriptor],
    ) -> None:
        self._sqw_io = sqw_io
        self._file_header = file_header
        self._data_block_descriptors = data_block_descriptors

    @classmethod
    @contextmanager
    def open(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        byteorder: Byteorder | Literal["little", "big"] | None = None,
    ) -> Iterator[SQW]:
        with open_binary(path, 'rb') as f:
            stored_path = None if isinstance(path, BinaryIO | BytesIO) else Path(path)
            sqw_io = LowLevelSqw(
                f, path=stored_path, byteorder=Byteorder.parse(byteorder)
            )
            file_header = _read_file_header(sqw_io)
            _ = sqw_io.read_u32()  # TODO don't know what this is
            data_block_descriptors = _read_data_block_descriptors(sqw_io)
            yield SQW(
                sqw_io=sqw_io,
                file_header=file_header,
                data_block_descriptors=data_block_descriptors,
            )

    @property
    def file_header(self) -> SqwFileHeader:
        return self._file_header

    @property
    def byteorder(self) -> Byteorder:
        return self._sqw_io.byteorder

    def get_data_block(self, name: tuple[str, str]) -> Any:
        try:
            block_descriptor = self._data_block_descriptors[name]
        except KeyError:
            raise KeyError(f"No data block {name!r} in file") from None

        self._sqw_io.seek(block_descriptor.position)
        return _squeeze(_read_objects(self._sqw_io))


def _read_objects(sqw_io: LowLevelSqw) -> Any:
    type_tag = sqw_io.read_u8()
    n_dims = sqw_io.read_u8()
    if type_tag == TypeTag.char.value:
        # Special case to properly decode string
        return _read_char_arrays(sqw_io, n_dims)

    shape = tuple(sqw_io.read_u32() for _ in range(n_dims))

    try:
        reader = _READERS_PER_TYPE[type_tag]
    except KeyError:
        raise NotImplementedError(f"No reader for SQW type {type_tag}") from None
    return _read_nd_object_array(sqw_io, shape, reader)


def _read_nd_object_array(
    sqw_io: LowLevelSqw, shape: tuple[int, ...], reader
) -> list[Any]:
    if len(shape) == 1:
        return [reader(sqw_io) for _ in range(shape[0])]
    return [_read_nd_object_array(sqw_io, shape[1:], reader) for _ in range(shape[0])]


def _read_char_arrays(sqw_io: LowLevelSqw, n_dims: int) -> str:
    if n_dims == 0:
        return ""
    if n_dims != 1:
        raise NotImplementedError(
            f"Cannot read char arrays with more than one dimension, got {n_dims}"
        )
    return sqw_io.read_char_array()


def _read_cell(sqw_io: LowLevelSqw) -> Any:
    return _read_objects(sqw_io)


def _read_struct(sqw_io: LowLevelSqw) -> Any:
    n_fields = sqw_io.read_u32()
    field_name_sizes = [sqw_io.read_u32() for _ in range(n_fields)]
    field_names = [sqw_io.read_n_chars(size) for size in field_name_sizes]
    field_values = _read_objects(sqw_io)
    field_values = _squeeze(field_values)

    return dataclasses.make_dataclass(
        _struct_serial_name(field_names, field_values),
        (
            _make_struct_field(name, value)
            for name, value in zip(field_names, field_values, strict=True)
        ),
        frozen=True,
        slots=True,
    )()


def _make_struct_field(name: str, value: Any) -> tuple[str, Any, dataclasses.Field]:
    return name, type(value), dataclasses.field(default_factory=lambda: value)


def _struct_serial_name(field_names: list[str], field_values: list[Any]) -> str:
    try:
        return str(
            next(
                iter(
                    value
                    for name, value in zip(field_names, field_values, strict=True)
                    if name == 'serial_name'
                )
            )
        )
    except StopIteration:
        return "unknown"


_READERS_PER_TYPE = {
    TypeTag.logical.value: LowLevelSqw.read_logical,
    TypeTag.f64.value: LowLevelSqw.read_f64,
    TypeTag.cell.value: _read_cell,
    TypeTag.struct.value: _read_struct,
}


def _read_file_header(sqw_io: LowLevelSqw) -> SqwFileHeader:
    prog_name = sqw_io.read_char_array()
    prog_version = sqw_io.read_f64()
    if prog_name != "horace" or prog_version != 4.0:
        warnings.warn(
            f"SQW program not supported: '{prog_name}' version {prog_version} "
            f"(expected 'horace' with version 4.0)",
            UserWarning,
            stacklevel=2,
        )

    sqw_type = SqwFileType(sqw_io.read_u32())
    n_dims = sqw_io.read_u32()
    return SqwFileHeader(
        prog_name=prog_name, prog_version=prog_version, sqw_type=sqw_type, n_dims=n_dims
    )


def _read_data_block_descriptors(
    sqw_io: LowLevelSqw,
) -> dict[tuple[str, str], SqwDataBlockDescriptor]:
    n_blocks = sqw_io.read_u32()
    descriptors = (_read_data_block_descriptor(sqw_io) for _ in range(n_blocks))
    return {descriptor.name: descriptor for descriptor in descriptors}


def _read_data_block_descriptor(sqw_io: LowLevelSqw) -> SqwDataBlockDescriptor:
    block_type = SqwDataBlockType(sqw_io.read_char_array())
    name = sqw_io.read_char_array(), sqw_io.read_char_array()
    position = sqw_io.read_u64()
    size = sqw_io.read_u32()
    locked = sqw_io.read_u32() == 1
    return SqwDataBlockDescriptor(
        block_type=block_type, name=name, position=position, size=size, locked=locked
    )


def _squeeze(nd_list: list[_T] | _T) -> list[_T] | _T:
    if isinstance(nd_list, list):
        if len(nd_list) == 1:
            return _squeeze(nd_list[0])
        return [_squeeze(item) for item in nd_list]
    return nd_list
