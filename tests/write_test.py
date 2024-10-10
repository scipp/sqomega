# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Literal

import pytest
import scipp as sc
import scipp.testing

from sqomega import Byteorder, EnergyMode, Sqw, SqwIXExperiment


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
        b"\x06\x00\x00\x00"
        b"horace"
        b"\x00\x00\x00\x00\x00\x00\x10\x40"
        b"\x01\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    assert buffer.read(len(expected)) == expected


def test_create_writes_file_header_big_endian() -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder="big")
    with builder.create():
        pass

    buffer.seek(0)
    expected = (
        b"\x00\x00\x00\x06"
        b"horace"
        b"\x40\x10\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x00"
    )
    assert buffer.read(len(expected)) == expected


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_create_writes_main_header(
    byteorder: Literal["native", "little", "big"],
) -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, title="my title", byteorder=byteorder)
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        main_header = sqw.read_data_block(("", "main_header"))
    assert main_header.full_filename == ""  # because we use a buffer
    assert main_header.title == "my title"
    assert main_header.nfiles == 0
    assert (main_header.creation_date - datetime.now(tz=timezone.utc)) < timedelta(
        seconds=1
    )


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_register_pixel_data_writes_pix_metadata(
    byteorder: Literal["native", "little", "big"],
) -> None:
    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder=byteorder)
    builder = builder.register_pixel_data(n_pixels=13, n_dims=3, experiments=[])
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        pix_metadata = sqw.read_data_block(("pix", "metadata"))
    assert pix_metadata.full_filename == ""  # because we use a buffer
    assert pix_metadata.npix == 13
    assert pix_metadata.data_range.shape == (9, 2)


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_writes_expdata(
    byteorder: Literal["native", "little", "big"],
) -> None:
    experiments = [
        SqwIXExperiment(
            run_id=0,
            efix=sc.scalar(1.2, unit='meV'),
            emode=EnergyMode.direct,
            en=sc.array(dims=['energy_transfer'], values=[3.0], unit='ueV'),
            psi=sc.scalar(1.2, unit='rad'),
            u=sc.vector([0.0, 1.0, 0.0]),
            v=sc.vector([1.0, 1.0, 0.0]),
            omega=sc.scalar(1.4, unit='rad'),
            dpsi=sc.scalar(46, unit='deg'),
            gl=sc.scalar(3, unit='rad'),
            gs=sc.scalar(-0.5, unit='rad'),
            filename="run1.nxspe",
            filepath='/data',
        ),
        SqwIXExperiment(
            run_id=2,
            efix=sc.scalar(0.16, unit='eV'),
            emode=EnergyMode.direct,
            en=sc.array(dims=['energy_transfer'], values=[2.0, 4.5], unit='meV'),
            psi=sc.scalar(-10.0, unit='deg'),
            u=sc.vector([1.0, 0.0, 0.0]),
            v=sc.vector([0.0, 1.0, 0.0]),
            omega=sc.scalar(-91, unit='deg'),
            dpsi=sc.scalar(-0.5, unit='rad'),
            gl=sc.scalar(0.0, unit='deg'),
            gs=sc.scalar(-5, unit='deg'),
            filename="run2.nxspe",
            filepath='/data',
        ),
    ]
    # The same as above but with canonical units.
    expected_experiments = [
        SqwIXExperiment(
            run_id=0,
            efix=sc.scalar(1.2, unit='meV'),
            emode=EnergyMode.direct,
            en=sc.array(dims=['energy_transfer'], values=[0.003], unit='meV'),
            psi=sc.scalar(1.2, unit='rad'),
            u=sc.vector([0.0, 1.0, 0.0]),
            v=sc.vector([1.0, 1.0, 0.0]),
            omega=sc.scalar(1.4, unit='rad'),
            dpsi=sc.scalar(46.0, unit='deg').to(unit='rad'),
            gl=sc.scalar(3.0, unit='rad'),
            gs=sc.scalar(-0.5, unit='rad'),
            filename="run1.nxspe",
            filepath='/data',
        ),
        SqwIXExperiment(
            run_id=2,
            efix=sc.scalar(160.0, unit='meV'),
            emode=EnergyMode.direct,
            en=sc.array(dims=['energy_transfer'], values=[2.0, 4.5], unit='meV'),
            psi=sc.scalar(-10.0, unit='deg').to(unit='rad'),
            u=sc.vector([1.0, 0.0, 0.0]),
            v=sc.vector([0.0, 1.0, 0.0]),
            omega=sc.scalar(-91.0, unit='deg').to(unit='rad'),
            dpsi=sc.scalar(-0.5, unit='rad'),
            gl=sc.scalar(0.0, unit='deg').to(unit='rad'),
            gs=sc.scalar(-5.0, unit='deg').to(unit='rad'),
            filename="run2.nxspe",
            filepath='/data',
        ),
    ]

    buffer = BytesIO()
    builder = Sqw.build(buffer, byteorder=byteorder)
    builder = builder.register_pixel_data(
        n_pixels=13, n_dims=3, experiments=experiments
    )
    with builder.create():
        pass
    buffer.seek(0)

    with Sqw.open(buffer) as sqw:
        loaded_experiments = sqw.read_data_block(("experiment_info", "expdata"))

    for loaded, expected in zip(loaded_experiments, expected_experiments, strict=True):
        for field in dataclasses.fields(expected):
            sc.testing.assert_identical(
                getattr(loaded, field.name), getattr(expected, field.name)
            )
