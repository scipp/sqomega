# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from io import BytesIO

import pytest

from sqomega import SQW, Byteorder, SqwFileHeader, SqwFileType

# TODO actual files in filesystem


def test_detects_byteorder_little_endian() -> None:
    buf = BytesIO(
        b'\x06\x00\x00\x00'
        b'horace'
        b'\x00\x00\x00\x00\x00\x00\x10\x40'
        b'\x01\x00\x00\x00'
        b'\x04\x00\x00\x00'
    )
    with SQW.open(buf) as sqw:
        assert sqw.byteorder == Byteorder.little


def test_detects_byteorder_big_endian() -> None:
    buf = BytesIO(
        b'\x00\x00\x00\x06'
        b'horace'
        b'\x40\x10\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x01'
        b'\x00\x00\x00\x04'
    )
    with SQW.open(buf) as sqw:
        assert sqw.byteorder == Byteorder.big


def test_open_file_header_little_endian() -> None:
    buf = BytesIO(
        b'\x06\x00\x00\x00'
        b'horace'
        b'\x00\x00\x00\x00\x00\x00\x10\x40'
        b'\x01\x00\x00\x00'
        b'\x04\x00\x00\x00'
    )
    expected = SqwFileHeader(
        prog_name="horace",
        prog_version=4.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with SQW.open(buf) as sqw:
        assert sqw.file_header == expected


def test_open_file_header_big_endian() -> None:
    buf = BytesIO(
        b'\x00\x00\x00\x06'
        b'horace'
        b'\x40\x10\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x01'
        b'\x00\x00\x00\x04'
    )
    expected = SqwFileHeader(
        prog_name="horace",
        prog_version=4.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with SQW.open(buf) as sqw:
        assert sqw.file_header == expected


def test_open_flags_wrong_prog_name() -> None:
    buf = BytesIO(
        b'\x07\x00\x00\x00'
        b'sqomega'
        b'\x00\x00\x00\x00\x00\x00\x10\x40'
        b'\x01\x00\x00\x00'
        b'\x04\x00\x00\x00'
    )
    expected = SqwFileHeader(
        prog_name="sqomega",
        prog_version=4.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with pytest.warns(UserWarning, match="SQW program not supported"):
        with SQW.open(buf) as sqw:
            assert sqw.file_header == expected


def test_open_flags_wrong_prog_version() -> None:
    buf = BytesIO(
        b'\x06\x00\x00\x00'
        b'horace'
        b'\x00\x00\x00\x00\x00\x00\x20\x40'
        b'\x01\x00\x00\x00'
        b'\x04\x00\x00\x00'
    )
    expected = SqwFileHeader(
        prog_name="horace",
        prog_version=8.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with pytest.warns(UserWarning, match="SQW program not supported"):
        with SQW.open(buf) as sqw:
            assert sqw.file_header == expected
