# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import functools
import struct
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, ParamSpec, TypeVar

from ._bytes import Byteorder

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _annotate_read_exception(
    ty: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Add a note with file-information to exceptions from read_* functions."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except ValueError as exc:
                sqw_io: LowLevelSqw = args[0]  # type: ignore[assignment]
                _add_note_to_read_exception(exc, sqw_io, ty)
                raise

        return wrapper

    return decorator


def _add_note_to_read_exception(exc: Exception, sqw_io: LowLevelSqw, ty: str) -> None:
    path_piece = (
        "in-memory SQW file" if sqw_io.path is None else f"SQW file '{sqw_io.path}'"
    )
    _add_note(
        exc,
        f"When reading a {ty} from {path_piece} at position {sqw_io.position}",
    )


def _add_note(exc: Exception, note: str) -> None:
    try:
        exc.add_note(note)  # type: ignore[attr-defined]
    except AttributeError:
        # Python < 3.11 -> do nothing and accept throwing a worse error.
        pass


class LowLevelSqw:
    def __init__(
        self, file: BinaryIO, *, path: Path | None, byteorder: Byteorder | None = None
    ) -> None:
        self._file = file
        self._byteorder = _deduce_byteorder(self._file, byteorder=byteorder)
        self._path = path

    @_annotate_read_exception("u32")
    def read_u32(self) -> int:
        buf = self._file.read(4)
        return int.from_bytes(buf, self._byteorder.get())

    @_annotate_read_exception("f64")
    def read_f64(self) -> float:
        buf = self._file.read(8)
        match self._byteorder:
            case Byteorder.little:
                bo = "<"
            case Byteorder.big:
                bo = ">"
        return struct.unpack(bo + "d", buf)[0]  # type: ignore[no-any-return]

    @_annotate_read_exception("char array")
    def read_char_array(self) -> str:
        size = self.read_u32()
        return self._file.read(size).decode('utf-8')

    @property
    def byteorder(self) -> Byteorder:
        return self._byteorder

    @property
    def position(self) -> int:
        return self._file.tell()

    @property
    def path(self) -> Path | None:
        return self._path


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
