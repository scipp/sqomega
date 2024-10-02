# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import struct
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from typing import BinaryIO, Literal

from ._bytes import Byteorder
from ._files import open_binary
from ._models import SQWFileHeader, SQWFileType


class SQWReader:
    def __init__(self, file: BinaryIO, *, byteorder: Byteorder | None = None) -> None:
        self._file = file
        self._byteorder, self._file_header = _read_header(
            self._file, byteorder=byteorder
        )

    @classmethod
    @contextmanager
    def open(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        byteorder: Byteorder | Literal["little", "big"] | None = None,
    ) -> Iterator[SQWReader]:
        with open_binary(path, 'rb') as f:
            yield cls(f, byteorder=Byteorder.parse(byteorder))


def _read_header(
    file: BinaryIO, *, byteorder: Byteorder | None = None
) -> tuple[Byteorder, SQWFileHeader]:
    byteorder, prog_name_size = _deduce_byteorder(file, byteorder=byteorder)
    prog_name = file.read(prog_name_size).decode('utf-8')
    prog_version = _double_from_bytes(file.read(8), byteorder)
    sqw_type = SQWFileType(int.from_bytes(file.read(4), byteorder.get()))
    n_dims = int.from_bytes(file.read(4), byteorder.get())
    return byteorder, SQWFileHeader(
        prog_name=prog_name, prog_version=prog_version, sqw_type=sqw_type, n_dims=n_dims
    )


def _double_from_bytes(b: bytes, byteorder: Byteorder) -> float:
    match byteorder:
        case Byteorder.little:
            bo = "<"
        case Byteorder.big:
            bo = ">"
    return struct.unpack(bo + "d", b)[0]  # type: ignore[no-any-return]


def _deduce_byteorder(
    file: BinaryIO, *, byteorder: Byteorder | None = None
) -> tuple[Byteorder, int]:
    """Guess the byte order of a file.

    The first four bytes of an SQW file are the length of the program name.
    Realistic lengths should be less than 2^16 bytes which is the flip over point
    between little and big endian.
    So we simply use the smaller number.
    """
    buf = file.read(4)

    if byteorder is not None:
        return byteorder, int.from_bytes(buf, byteorder.get())

    le_size = int.from_bytes(buf, 'little')
    be_size = int.from_bytes(buf, 'big')
    if le_size < be_size:
        return Byteorder.little, le_size
    return Byteorder.big, be_size
