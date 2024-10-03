# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar


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
class SqwMainHeader:
    full_filename: str
    title: str
    nfiles: int  # f64
    creation_date: datetime  # char_array

    serial_name: ClassVar[str] = "main_header_cl"
    version: ClassVar[float] = 2.0
