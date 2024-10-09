# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import numpy.typing as npt
from dateutil.parser import parse as parse_datetime

from . import _ir as ir
from ._build import SqwBuilder
from ._bytes import Byteorder
from ._files import open_binary
from ._low_level_io import LowLevelSqw
from ._models import (
    DataBlockName,
    SqwDataBlockDescriptor,
    SqwDataBlockType,
    SqwFileHeader,
    SqwFileType,
    SqwMainHeader,
    SqwPixMetadata,
)
from ._read_write import read_object_array


class Sqw:
    def __init__(
        self,
        *,
        sqw_io: LowLevelSqw,
        file_header: SqwFileHeader,
        block_allocation_table: dict[DataBlockName, SqwDataBlockDescriptor],
    ) -> None:
        self._sqw_io = sqw_io
        self._file_header = file_header
        self._block_allocation_table = block_allocation_table

    @classmethod
    @contextmanager
    def open(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        byteorder: Byteorder | Literal["little", "big"] | None = None,
    ) -> Generator[Sqw, None, None]:
        with open_binary(path, 'rb') as f:
            stored_path = None if isinstance(path, BinaryIO | BytesIO) else Path(path)
            sqw_io = LowLevelSqw(
                f,
                path=stored_path,
                byteorder=Byteorder.parse(byteorder) if byteorder is not None else None,
            )
            file_header = _read_file_header(sqw_io)
            _descriptors_size = sqw_io.read_u32()  # don't need this
            data_block_descriptors = _read_data_block_descriptors(sqw_io)
            yield Sqw(
                sqw_io=sqw_io,
                file_header=file_header,
                block_allocation_table=data_block_descriptors,
            )

    @classmethod
    def build(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        title: str = '',
        byteorder: Byteorder | Literal["native", "little", "big"] = "native",
    ) -> SqwBuilder:
        return SqwBuilder(path, title, byteorder=Byteorder.parse(byteorder))

    @property
    def file_header(self) -> SqwFileHeader:
        return self._file_header

    @property
    def byteorder(self) -> Byteorder:
        return self._sqw_io.byteorder

    def read_data_block(self, name: DataBlockName) -> Any:  # TODO type
        try:
            block_descriptor = self._block_allocation_table[name]
        except KeyError:
            raise KeyError(f"No data block {name!r} in file") from None

        self._sqw_io.seek(block_descriptor.position)
        match block_descriptor.block_type:
            case SqwDataBlockType.regular:
                return _parse_block(read_object_array(self._sqw_io))
            case SqwDataBlockType.pix:
                return _read_pix_block(self._sqw_io)
            case _:
                raise NotImplementedError(
                    f"Unsupported data block type: {block_descriptor.block_type}"
                )


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
    if sqw_type != SqwFileType.SQW:
        warnings.warn("DND files are not supported", UserWarning, stacklevel=2)

    n_dims = sqw_io.read_u32()
    return SqwFileHeader(
        prog_name=prog_name, prog_version=prog_version, sqw_type=sqw_type, n_dims=n_dims
    )


def _read_data_block_descriptors(
    sqw_io: LowLevelSqw,
) -> dict[DataBlockName, SqwDataBlockDescriptor]:
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


class AbortParse(Exception): ...


def _parse_block(block: ir.ObjectArray | ir.CellArray) -> Any:
    try:
        if isinstance(block, ir.CellArray):
            raise AbortParse("Block is a cell array, not a struct")
        return _try_parse_block(block)
    except AbortParse as abort:
        warnings.warn(
            f"Unable to parse SQW block: {abort.args[0]}", UserWarning, stacklevel=2
        )
        return block


def _try_parse_block(block: ir.ObjectArray) -> Any:
    if block.shape != (1,):
        raise AbortParse(f"Unsupported block shape: {block.shape}")
    if block.ty != ir.TypeTag.struct:
        raise AbortParse(f"Unsupported block type: {block.shape}, expected struct")
    struct: ir.Struct = block.data[0]  # type: ignore[assignment]
    if sum(s != 1 for s in struct.field_values.shape) != 1:
        raise AbortParse(
            "Contents cannot be a multi-dimensional cell array, "
            f"got shape {struct.field_values.shape}"
        )

    key = _get_struct_type_id(struct)
    try:
        parser = _BLOCK_PARSERS[key]
    except KeyError:
        raise AbortParse(f"No parser for struct type {key}") from None
    return parser(struct)


def _get_struct_type_id(struct: ir.Struct) -> tuple[str, float]:
    name = _get_struct_field(struct, 'serial_name')
    if len(name.shape) != 1:
        raise AbortParse("'serial_name' is multi-dimensional")
    version = _get_struct_field(struct, 'version')
    if version.shape != (1,):
        raise AbortParse("'version' is multi-dimensional")

    n = name.data[0]
    v = version.data[0]
    if not isinstance(n, ir.String):
        raise AbortParse("'serial_name' is not a string")
    if not isinstance(v, ir.F64):
        raise AbortParse("'version' is not an f64")

    return n.value, v.value


def _get_struct_field(struct: ir.Struct, name: str) -> ir.ObjectArray:
    for field_name, value in zip(
        struct.field_names, struct.field_values.data, strict=True
    ):
        if field_name == name:
            return value
    raise AbortParse(f"No field '{name}' in struct")


def _get_scalar_struct_field(struct: ir.Struct, name: str) -> Any:
    field = _get_struct_field(struct, name)
    shape = field.shape[1:] if field.ty == ir.TypeTag.char else field.shape
    if shape not in ((1,), ()):
        raise AbortParse(f"Field '{name}' has non-scalar shape: {shape}")
    if isinstance(field.data[0], ir.Struct):
        raise AbortParse(f"Field '{name}' contains a nested struct")
    return field.data[0].value


def _parse_main_header_cl_2_0(struct: ir.Struct) -> SqwMainHeader:
    return SqwMainHeader(
        full_filename=_get_scalar_struct_field(struct, 'full_filename'),
        title=_get_scalar_struct_field(struct, 'title'),
        nfiles=int(_get_scalar_struct_field(struct, 'nfiles')),
        creation_date=parse_datetime(_get_scalar_struct_field(struct, 'creation_date')),
    )


def _parse_pix_metadata_1_0(struct: ir.Struct) -> SqwPixMetadata:
    data_range = _get_struct_field(struct, 'data_range').data
    if not isinstance(data_range, np.ndarray):
        raise AbortParse("'data_range' is not a numpy array")
    return SqwPixMetadata(
        full_filename=_get_scalar_struct_field(struct, 'full_filename'),
        npix=int(_get_scalar_struct_field(struct, 'npix')),
        data_range=data_range,
    )


_BLOCK_PARSERS = {
    (SqwMainHeader.serial_name, SqwMainHeader.version): _parse_main_header_cl_2_0,
    (SqwPixMetadata.serial_name, SqwPixMetadata.version): _parse_pix_metadata_1_0,
}


def _read_pix_block(sqw_io: LowLevelSqw) -> npt.NDArray[np.float32]:
    n_rows = sqw_io.read_u32()
    n_pixels = sqw_io.read_u64()
    return sqw_io.read_array((n_rows, n_pixels), np.dtype('float32'))
