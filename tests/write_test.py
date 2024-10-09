# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Literal

import pytest

from sqomega import Byteorder, Sqw


def test_create_sets_byteorder_native() -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer)
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        assert sqw.byteorder.value == sys.byteorder


def test_create_sets_byteorder_little() -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder="little")
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        assert sqw.byteorder == Byteorder.little


def test_create_sets_byteorder_big() -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder="big")
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        assert sqw.byteorder == Byteorder.big


def test_create_writes_file_header_little_endian() -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder="little")
    with builder.create():
        pass

    buffer.seek(0)
    expected = (
        b'\x06\x00\x00\x00'
        b'horace'
        b'\x00\x00\x00\x00\x00\x00\x10\x40'
        b'\x01\x00\x00\x00'
        b'\x00\x00\x00\x00'
    )
    assert buffer.read(len(expected)) == expected


def test_create_writes_file_header_big_endian() -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder="big")
    with builder.create():
        pass

    buffer.seek(0)
    expected = (
        b'\x00\x00\x00\x06'
        b'horace'
        b'\x40\x10\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x01'
        b'\x00\x00\x00\x00'
    )
    assert buffer.read(len(expected)) == expected


@pytest.mark.parametrize('byteorder', ['native', 'little', 'big'])
def test_create_writes_main_header(
    byteorder: Literal['native', 'little', 'big'],
) -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, title='my title', byteorder=byteorder)
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        main_header = sqw.read_data_block(('', 'main_header'))
    assert main_header.full_filename == ''  # because we use a buffer
    assert main_header.title == 'my title'
    assert main_header.nfiles == 0
    assert (main_header.creation_date - datetime.now(tz=timezone.utc)) < timedelta(
        seconds=1
    )


@pytest.mark.parametrize('byteorder', ['native', 'little', 'big'])
def test_register_pixel_data_writes_pix_metadata(
    byteorder: Literal['native', 'little', 'big'],
) -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder=byteorder)
    builder = builder.register_pixel_data(n_pixels=13, n_dims=3)
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        pix_metadata = sqw.read_data_block(('pix', 'metadata'))
    assert pix_metadata.full_filename == ''  # because we use a buffer
    assert pix_metadata.npix == 13
    assert pix_metadata.data_range.shape == (9, 2)
