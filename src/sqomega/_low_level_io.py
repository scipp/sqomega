# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import struct
from typing import BinaryIO

from ._bytes import Byteorder

# TODO annotate exceptions


class LowLevelSqw:
    def __init__(self, file: BinaryIO, *, byteorder: Byteorder | None = None) -> None:
        self._file = file
        self._byteorder = _deduce_byteorder(self._file, byteorder=byteorder)

    @property
    def byteorder(self) -> Byteorder:
        return self._byteorder

    def read_u32(self) -> int:
        buf = self._file.read(4)
        return int.from_bytes(buf, self._byteorder.get())

    def read_f64(self) -> float:
        buf = self._file.read(8)
        match self._byteorder:
            case Byteorder.little:
                bo = "<"
            case Byteorder.big:
                bo = ">"
        return struct.unpack(bo + "d", buf)[0]  # type: ignore[no-any-return]

    def read_char_array(self) -> str:
        size = self.read_u32()
        return self._file.read(size).decode('utf-8')


def _deduce_byteorder(
    file: BinaryIO, *, byteorder: Byteorder | None = None
) -> Byteorder:
    """Guess the byte order of a file.

    The first four bytes of an SQW file are the length of the program name.
    Realistic lengths should be less than 2^16 bytes which is the flip over point
    between little and big endian.
    So we simply use the smaller number.

    This could be made more robust by reading (parts of) the rest of the header
    and checking that it makes sense, e.g., that the program name is valid UTF-8.
    But since HORACE ignores the issue of byteorder, what we have here should be enough.
    """
    if byteorder is not None:
        return byteorder

    pos = file.tell()
    buf = file.read(4)
    file.seek(pos)

    le_size = int.from_bytes(buf, 'little')
    be_size = int.from_bytes(buf, 'big')
    if le_size < be_size:
        return Byteorder.little
    return Byteorder.big
