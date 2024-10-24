# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from contextlib import AbstractContextManager, nullcontext
from io import BytesIO
from os import PathLike
from typing import BinaryIO, Literal


def open_binary(
    path: str | PathLike[str] | BytesIO | BinaryIO, mode: Literal["rb", "wb", "r+b"]
) -> AbstractContextManager[BinaryIO]:
    """Open a binary file at a path or return an already open file."""
    if isinstance(path, BytesIO | BinaryIO):
        return nullcontext(path)
    return open(path, mode)
