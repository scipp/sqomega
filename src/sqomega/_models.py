# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import enum
from dataclasses import dataclass


class SQWFileType(enum.Enum):
    DND = 0
    SQW = 1


@dataclass(frozen=True, kw_only=True, slots=True)
class SQWFileHeader:
    prog_name: str
    prog_version: float
    sqw_type: SQWFileType
    n_dims: int
