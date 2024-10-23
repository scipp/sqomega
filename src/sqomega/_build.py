# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import dataclasses
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np

from ._bytes import Byteorder
from ._files import open_binary
from ._low_level_io import LowLevelSqw
from ._models import (
    DataBlockName,
    SqwDataBlockDescriptor,
    SqwDataBlockType,
    SqwDndMetadata,
    SqwFileHeader,
    SqwFileType,
    SqwIXExperiment,
    SqwMainHeader,
    SqwMultiIXExperiment,
    SqwPixelMetadata,
)
from ._read_write import write_object_array

if TYPE_CHECKING:
    from ._sqw import Sqw

# Based on
# https://github.com/pace-neutrons/Horace/blob/master/documentation/add/05_file_formats.md
_DEFAULT_PIX_ROWS = (
    "u1",  # Coordinate 1 (h)
    "u2",  # Coordinate 2 (k)
    "u3",  # Coordinate 3 (l)
    "u4",  # Coordinate 1 (E)
    "irun",  # Run index in header block
    "idet",  # Detector group number
    "ien",  # Energy bin number
    "signal",  # Signal
    "error",  # Variance
)


class SqwBuilder:
    def __init__(
        self,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        title: str,
        *,
        byteorder: Byteorder,
    ) -> None:
        self._path = path
        self._stored_path = (
            None if isinstance(self._path, BinaryIO | BytesIO) else Path(self._path)
        )
        self._byteorder = byteorder
        self._n_dim = 0

        main_header = SqwMainHeader(
            full_filename=self._full_filename,
            title=title,
            nfiles=0,
            # To be replaced when writing the file.
            creation_date=datetime(1, 1, 1, tzinfo=timezone.utc),
        )
        self._data_blocks: dict[DataBlockName, Any] = {  # TODO type
            ("", "main_header"): main_header,
        }

        self._pix_placeholder: _PixPlaceholder | None = None
        self._expdata: list[SqwIXExperiment] = []

    @contextmanager
    def create(self) -> Generator[Sqw, None, None]:
        from ._sqw import Sqw

        with open_binary(self._path, "wb") as f:
            sqw_io = LowLevelSqw(
                f,
                path=self._stored_path,
                byteorder=self._byteorder,
            )

            file_header = self._make_file_header()
            _write_file_header(sqw_io, file_header)

            block_buffers, block_descriptors = self._serialize_data_blocks()
            bat_buffer, block_descriptors = self._serialize_block_allocation_table(
                block_descriptors=block_descriptors,
                bat_offset=sqw_io.position,
            )
            sqw_io.write_raw(bat_buffer)
            for name, buffer in block_buffers.items():
                descriptor = block_descriptors[name]
                match descriptor.block_type:
                    case SqwDataBlockType.regular:
                        sqw_io.write_raw(buffer)
                    case SqwDataBlockType.pix:
                        self._pix_placeholder.write(sqw_io)
                        sqw_io.seek(descriptor.position + descriptor.size)
                    case _:
                        raise NotImplementedError(
                            f"Unsupported data block type: {descriptor.block_type}"
                        )

            yield Sqw(sqw_io=sqw_io, file_header=file_header, block_allocation_table={})

    def register_pixel_data(
        self,
        *,
        n_pixels: int,
        n_dims: int,
        experiments: list[SqwIXExperiment],
        rows: tuple[str, ...] = _DEFAULT_PIX_ROWS,
    ) -> SqwBuilder:
        if self._pix_placeholder is not None:
            raise RuntimeError("SQW builder already has pixel data")
        self._n_dim = n_dims

        self._expdata = experiments
        self._data_blocks[('experiment_info', 'expdata')] = SqwMultiIXExperiment(
            self._expdata
        )
        self._data_blocks[("pix", "metadata")] = SqwPixelMetadata(
            full_filename=self._full_filename,
            npix=n_pixels,
            data_range=np.c_[np.full(len(rows), np.inf), np.full(len(rows), -np.inf)],
        )
        self._pix_placeholder = _PixPlaceholder(
            n_pixels=n_pixels,
            rows=rows,
        )
        return self

    def add_dnd_metadata(self, block: SqwDndMetadata) -> SqwBuilder:
        self._data_blocks[("data", "metadata")] = block
        return self

    def _make_file_header(self) -> SqwFileHeader:
        return SqwFileHeader(
            prog_name="horace",
            prog_version=4.0,
            sqw_type=SqwFileType.SQW,
            n_dims=self._n_dim,
        )

    def _serialize_data_blocks(
        self,
    ) -> tuple[
        dict[DataBlockName, memoryview | None],
        dict[DataBlockName, SqwDataBlockDescriptor],
    ]:
        data_blocks = self._prepare_data_blocks()
        buffers = {}
        descriptors = {}
        for name, data_block in data_blocks.items():
            buffer = BytesIO()
            sqw_io = LowLevelSqw(
                buffer, path=self._stored_path, byteorder=self._byteorder
            )
            write_object_array(sqw_io, data_block.serialize_to_ir().to_object_array())

            buffer.seek(0)
            buf = buffer.getbuffer()
            buffers[name] = buf
            descriptors[name] = SqwDataBlockDescriptor(
                block_type=SqwDataBlockType.regular,
                name=name,
                position=0,
                size=len(buf),
                locked=False,
            )

        if self._pix_placeholder is not None:
            buffers[('pix', 'data_wrap')] = None
            descriptors[('pix', 'data_wrap')] = SqwDataBlockDescriptor(
                block_type=SqwDataBlockType.pix,
                name=('pix', 'data_wrap'),
                position=0,
                size=4 + 8 + self._pix_placeholder.pix_array_size(),
                locked=False,
            )

        return buffers, descriptors

    def _prepare_data_blocks(self) -> dict[DataBlockName, Any]:
        filepath, filename = self._filepath_and_name
        return {
            key: block.prepare_for_serialization(filepath=filepath, filename=filename)
            for key, block in self._data_blocks.items()
        }

    def _serialize_block_allocation_table(
        self,
        block_descriptors: dict[DataBlockName, SqwDataBlockDescriptor],
        bat_offset: int,
    ) -> tuple[memoryview, dict[DataBlockName, SqwDataBlockDescriptor]]:
        # This function first writes the block allocation table (BAT) with placeholder
        # values in order to determine the size of the BAT.
        # Then, it computes the actual positions that data blocks will have in the file
        # and inserts those positions into the serialized BAT.
        # It returns a buffer of the BAT that can be inserted right after the file
        # header and an updated in-memory representation of the BAT.

        buffer = BytesIO()
        sqw_io = LowLevelSqw(buffer, path=self._stored_path, byteorder=self._byteorder)
        sqw_io.write_u32(0)  # Size of BAT in bytes, filled in below.
        sqw_io.write_u32(len(block_descriptors))
        bat_begin = sqw_io.position
        # Offsets are relative to the local sqw_io.
        position_offsets = {
            name: _write_data_block_descriptor(sqw_io, descriptor)
            for name, descriptor in block_descriptors.items()
        }
        bat_size = sqw_io.position - bat_begin

        block_position = bat_offset + sqw_io.position
        amended_descriptors = {}
        for name, descriptor in block_descriptors.items():
            amended_descriptors[name] = dataclasses.replace(
                descriptor, position=block_position
            )
            offset = position_offsets[name]
            sqw_io.seek(offset)
            sqw_io.write_u64(block_position)
            block_position += descriptor.size

        sqw_io.seek(0)
        sqw_io.write_u32(bat_size)
        return buffer.getbuffer(), amended_descriptors

    @property
    def _full_filename(self) -> str:
        return os.fspath(self._stored_path or "")

    @property
    def _filepath_and_name(self) -> tuple[str, str]:
        if self._stored_path is None:
            return '', ''
        return os.fspath(self._stored_path.parent), self._stored_path.name


def _write_file_header(sqw_io: LowLevelSqw, file_header: SqwFileHeader) -> None:
    sqw_io.write_char_array(file_header.prog_name)
    sqw_io.write_f64(file_header.prog_version)
    sqw_io.write_u32(file_header.sqw_type.value)
    sqw_io.write_u32(file_header.n_dims)


def _write_data_block_descriptor(
    sqw_io: LowLevelSqw, descriptor: SqwDataBlockDescriptor
) -> int:
    sqw_io.write_char_array(descriptor.block_type.value)
    sqw_io.write_char_array(descriptor.name[0])
    sqw_io.write_char_array(descriptor.name[1])
    pos = sqw_io.position
    sqw_io.write_u64(descriptor.position)
    sqw_io.write_u32(descriptor.size)
    sqw_io.write_u32(int(descriptor.locked))
    return pos


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class _PixPlaceholder:
    n_pixels: int
    rows: tuple[str, ...]

    def pix_array_size(self) -> int:
        # *4 for f32
        return self.n_pixels * len(self.rows) * 4

    def write(self, sqw_io: LowLevelSqw) -> None:
        sqw_io.write_u32(len(self.rows))
        sqw_io.write_u64(self.n_pixels)
