# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import ClassVar

import numpy as np

from . import _ir as ir

DataBlockName = tuple[str, str]


class SqwFileType(enum.Enum):
    DND = 0
    SQW = 1


@dataclass(frozen=True, kw_only=True, slots=True)
class SqwFileHeader:
    prog_name: str
    prog_version: float
    sqw_type: SqwFileType
    n_dims: int


class SqwDataBlockType(enum.Enum):
    regular = "data_block"
    pix = "pix_data_block"
    dnd = "dnd_data_block"


@dataclass(frozen=True, kw_only=True, slots=True)
class SqwDataBlockDescriptor:
    block_type: SqwDataBlockType
    name: DataBlockName
    position: int  # u64
    size: int  # u32
    locked: bool  # u32


@dataclass(kw_only=True, slots=True)
class SqwMainHeader(ir.Serializable):
    full_filename: str
    title: str
    nfiles: int
    creation_date: datetime

    serial_name: ClassVar[str] = "main_header_cl"
    version: ClassVar[float] = 2.0

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "full_filename": ir.String(self.full_filename),
            "title": ir.String(self.title),
            "nfiles": ir.F64(self.nfiles),
            "creation_date": ir.Datetime(self.creation_date),
            "creation_date_defined_privately": ir.Logical(False),
        }

    def prepare_for_serialization(self) -> SqwMainHeader:
        return replace(self, creation_date=datetime.now(tz=timezone.utc))


@dataclass(kw_only=True, slots=True)
class SqwPixMetadata(ir.Serializable):
    full_filename: str
    npix: int
    data_range: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    serial_name: ClassVar[str] = "pix_metadata"
    version: ClassVar[float] = 1.0

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "full_filename": ir.String(self.full_filename),
            "npix": ir.F64(self.npix),
            "data_range": ir.Array(self.data_range, ty=ir.TypeTag.f64),
        }


@dataclass(kw_only=True, slots=True)
class SqwPixWrap(ir.Serializable):
    """Represents pixel data but does not hold the actual data."""

    n_rows: int = 9
    n_pixels: int

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "n_rows": ir.U32(self.n_rows),
            "n_pixels": ir.U64(self.n_pixels),
        }
