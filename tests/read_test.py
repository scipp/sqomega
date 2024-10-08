# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest

from sqomega import Byteorder, Sqw, SqwFileHeader, SqwFileType, SqwMainHeader

# TODO actual files in filesystem


def test_detects_byteorder_little_endian() -> None:
    buf = BytesIO(
        b'\x06\x00\x00\x00'
        b'horace'
        b'\x00\x00\x00\x00\x00\x00\x10\x40'
        b'\x01\x00\x00\x00'
        b'\x04\x00\x00\x00'
    )
    with Sqw.open(buf) as sqw:
        assert sqw.byteorder == Byteorder.little


def test_detects_byteorder_big_endian() -> None:
    buf = BytesIO(
        b'\x00\x00\x00\x06'
        b'horace'
        b'\x40\x10\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x01'
        b'\x00\x00\x00\x04'
    )
    with Sqw.open(buf) as sqw:
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
    with Sqw.open(buf) as sqw:
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
    with Sqw.open(buf) as sqw:
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
        with Sqw.open(buf) as sqw:
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
        with Sqw.open(buf) as sqw:
            assert sqw.file_header == expected


# TODO use test file
@pytest.fixture
def intact_v4_sqw() -> Path:
    return Path(__file__).resolve().parent.parent / 'data' / 'fe_demo.sqw'


def test_read_main_header(intact_v4_sqw: Path) -> None:
    with Sqw.open(intact_v4_sqw) as sqw:
        main_header = sqw.read_data_block(('', 'main_header'))
    assert main_header.version == 2.0
    assert (
        main_header.full_filename
        == 'C:\\Users\\henrikjacobsen3\\Documents\\Horace\\demo\\fe_demo.sqw'
    )
    assert main_header.title == ''
    assert type(main_header.nfiles) is int  # because it is encoded as f64 in file
    assert main_header.nfiles == 23
    # TODO can we encode a timezone? How does horace react?
    assert main_header.creation_date == datetime(2024, 9, 16, 9, 32, 30)


# def test_read_detpar(intact_v4_sqw: Path) -> None:
#     with SQW.open(intact_v4_sqw) as sqw:
#         print(sqw._block_allocation_table[('', 'detpar')])
#         detpar = sqw.read_data_block(('', 'detpar'))
#         print(sqw._sqw_io.position)
#     # print(detpar)
#     with open('asd.txt', 'w') as f:
#         f.write(str(detpar))
#     assert False
