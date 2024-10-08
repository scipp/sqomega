# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from . import _ir as ir


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
    name: tuple[str, str]
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
