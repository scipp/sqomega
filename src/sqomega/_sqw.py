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

from dateutil.parser import parse as parse_datetime

from ._bytes import Byteorder
from ._files import open_binary
from ._low_level_io import LowLevelSqw
from ._models import (
    SqwDataBlockDescriptor,
    SqwDataBlockType,
    SqwFileHeader,
    SqwFileType,
    SqwMainHeader,
)
from ._read_write import read_objects


class SQW:
    def __init__(
        self,
        *,
        sqw_io: LowLevelSqw,
        file_header: SqwFileHeader,
        block_allocation_table: dict[tuple[str, str], SqwDataBlockDescriptor],
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
    ) -> Generator[SQW, None, None]:
        with open_binary(path, 'rb') as f:
            stored_path = None if isinstance(path, BinaryIO | BytesIO) else Path(path)
            sqw_io = LowLevelSqw(
                f, path=stored_path, byteorder=Byteorder.parse(byteorder)
            )
            file_header = _read_file_header(sqw_io)
            _descriptors_size = sqw_io.read_u32()  # don't need this
            data_block_descriptors = _read_data_block_descriptors(sqw_io)
            yield SQW(
                sqw_io=sqw_io,
                file_header=file_header,
                block_allocation_table=data_block_descriptors,
            )

    @property
    def file_header(self) -> SqwFileHeader:
        return self._file_header

    @property
    def byteorder(self) -> Byteorder:
        return self._sqw_io.byteorder

    def read_data_block(self, name: tuple[str, str]) -> Any:
        try:
            block_descriptor = self._block_allocation_table[name]
        except KeyError:
            raise KeyError(f"No data block {name!r} in file") from None

        # TODO branch on block type
        self._sqw_io.seek(block_descriptor.position)
        return _parse_block(read_objects(self._sqw_io))


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


def _parse_block(block: object) -> Any:
    key = (getattr(block, 'serial_name', None), getattr(block, 'version', None))
    try:
        parser = _BLOCK_PARSERS[key]
    except KeyError:
        return block
    return parser(block)


def _parse_main_header_cl_2_0(obj: Any) -> SqwMainHeader:
    return SqwMainHeader(
        full_filename=obj.full_filename,
        title=obj.title,
        nfiles=int(obj.nfiles),
        creation_date=parse_datetime(obj.creation_date),
    )


_BLOCK_PARSERS = {
    (SqwMainHeader.serial_name, SqwMainHeader.version): _parse_main_header_cl_2_0,
}
