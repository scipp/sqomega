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
    logical = b'\x00'
    char = b'\x01'
    double = b'\x03'
    int8 = b'\x05'
    uint8 = b'\x06'
    int32 = b'\x09'
    uint32 = b'\x0a'
    int64 = b'\x0b'
    uint64 = b'\x0c'
    cell = b'\x17'
    struct = b'\x18'
