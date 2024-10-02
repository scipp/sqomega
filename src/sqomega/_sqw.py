# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Literal

from ._bytes import Byteorder
from ._files import open_binary
from ._low_level_io import LowLevelSqw
from ._models import SqwFileHeader, SqwFileType


class SQW:
    def __init__(self, *, sqw_io: LowLevelSqw, file_header: SqwFileHeader) -> None:
        self._sqw_io = sqw_io
        self._file_header = file_header

    @classmethod
    @contextmanager
    def open(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        byteorder: Byteorder | Literal["little", "big"] | None = None,
    ) -> Iterator[SQW]:
        with open_binary(path, 'rb') as f:
            stored_path = None if isinstance(path, BinaryIO | BytesIO) else Path(path)
            sqw_io = LowLevelSqw(
                f, path=stored_path, byteorder=Byteorder.parse(byteorder)
            )
            file_header = _read_header(sqw_io)
            yield SQW(sqw_io=sqw_io, file_header=file_header)

    @property
    def file_header(self) -> SqwFileHeader:
        return self._file_header

    @property
    def byteorder(self) -> Byteorder:
        return self._sqw_io.byteorder


def _read_header(sqw_io: LowLevelSqw) -> SqwFileHeader:
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
    n_dims = sqw_io.read_u32()
    return SqwFileHeader(
        prog_name=prog_name, prog_version=prog_version, sqw_type=sqw_type, n_dims=n_dims
    )
