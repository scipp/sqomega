# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import enum
from typing import Literal


class Byteorder(enum.Enum):
    little = "little"
    big = "big"

    @classmethod
    def parse(
        cls, value: Byteorder | Literal["little", "big"] | None = None
    ) -> Byteorder | None:
        if value is None:
            return None
        if isinstance(value, Byteorder):
            return value
        if isinstance(value, str):
            return cls[value]
        raise ValueError(f"Invalid Byteorder: {value}")

    def get(self) -> Literal["little", "big"]:
        match self:
            case Byteorder.little:
                return "little"
            case Byteorder.big:
                return "big"


class TypeTag(enum.Enum):
    # Gaps in values are unsupported types.
    logical = 0
    char = 1
    f64 = 3
    f32 = 4
    i8 = 5
    u8 = 6
    i32 = 9
    u32 = 10
    i64 = 11
    u64 = 12
    cell = 23
    struct = 24
