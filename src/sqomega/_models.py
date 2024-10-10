# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import ClassVar

import numpy as np
import scipp as sc

from . import _ir as ir

DataBlockName = tuple[str, str]


class SqwFileType(enum.Enum):
    DND = 0
    SQW = 1


@dataclass(frozen=True, kw_only=True, slots=True)
class SqwFileHeader:
    prog_name: str
    prog_version: float
    sqw_type: SqwFileType
    n_dims: int


class SqwDataBlockType(enum.Enum):
    regular = "data_block"
    pix = "pix_data_block"
    dnd = "dnd_data_block"


@dataclass(frozen=True, kw_only=True, slots=True)
class SqwDataBlockDescriptor:
    block_type: SqwDataBlockType
    name: DataBlockName
    position: int  # u64
    size: int  # u32
    locked: bool  # u32


@dataclass(kw_only=True, slots=True)
class SqwMainHeader(ir.Serializable):
    full_filename: str
    title: str
    nfiles: int
    creation_date: datetime

    serial_name: ClassVar[str] = "main_header_cl"
    version: ClassVar[float] = 2.0

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "full_filename": ir.String(self.full_filename),
            "title": ir.String(self.title),
            "nfiles": ir.F64(float(self.nfiles)),
            "creation_date": ir.Datetime(self.creation_date),
            "creation_date_defined_privately": ir.Logical(False),
        }

    def prepare_for_serialization(self) -> SqwMainHeader:
        return replace(self, creation_date=datetime.now(tz=timezone.utc))


@dataclass(kw_only=True, slots=True)
class SqwPixelMetadata(ir.Serializable):
    full_filename: str
    npix: int
    data_range: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    serial_name: ClassVar[str] = "pix_metadata"
    version: ClassVar[float] = 1.0

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "full_filename": ir.String(self.full_filename),
            "npix": ir.F64(float(self.npix)),
            "data_range": ir.Array(self.data_range, ty=ir.TypeTag.f64),
        }


@dataclass(kw_only=True, slots=True)
class SqwPixWrap(ir.Serializable):
    """Represents pixel data but does not hold the actual data."""

    n_rows: int = 9
    n_pixels: int

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "n_rows": ir.U32(self.n_rows),
            "n_pixels": ir.U64(self.n_pixels),
        }


class EnergyMode(enum.Enum):
    direct = 1
    indirect = 2


# In contrast to SQW files, this model contains the nested
# struct fields instead of a nested struct in `array-dat`.
@dataclass(kw_only=True, slots=True)
class SqwIXExperiment(ir.Serializable):
    run_id: int
    # 1 element for direct, array of detector.shape for indirect
    efix: sc.Variable  # array or scalar
    emode: EnergyMode
    en: sc.Variable  # array
    psi: sc.Variable  # scalar
    u: sc.Variable  # vector
    v: sc.Variable  # vector
    omega: sc.Variable  # scalar
    dpsi: sc.Variable  # scalar
    gl: sc.Variable  # scalar
    gs: sc.Variable  # scalar
    filename: str = ""
    filepath: str = ""

    serial_name: ClassVar[str] = "IX_experiment"
    version: ClassVar[float] = 3.0

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        en = (
            self.en.to(unit='meV', dtype='float64', copy=False)
            .broadcast(sizes={'_': 1, 'energy_transfer': self.en.shape[0]})
            .values
        )
        efix = self.efix.to(unit='meV', dtype='float64', copy=False)
        if efix.ndim == 0:
            efix = efix.broadcast(sizes={'_': 1})
        return {
            "filename": ir.String(self.filename),
            "filepath": ir.String(self.filepath),
            "run_id": ir.F64(float(self.run_id)),
            "efix": ir.Array(efix.values, ty=ir.TypeTag.f64),
            "emode": ir.F64(float(self.emode.value)),
            "en": ir.Array(en, ty=ir.TypeTag.f64),
            "psi": ir.F64(_angle_value(self.psi)),
            "u": ir.Array(self.u.values, ty=ir.TypeTag.f64),
            "v": ir.Array(self.v.values, ty=ir.TypeTag.f64),
            "omega": ir.F64(_angle_value(self.omega)),
            "dpsi": ir.F64(_angle_value(self.dpsi)),
            "gl": ir.F64(_angle_value(self.gl)),
            "gs": ir.F64(_angle_value(self.gs)),
            "angular_is_degree": ir.Logical(False),
            # serial_name and version are serialized by SqwMultiIXExperiment
        }


def _angle_value(x: sc.Variable) -> float:
    return x.to(unit='rad', dtype='float64', copy=False).value


@dataclass(slots=True)
class SqwMultiIXExperiment(ir.Serializable):
    array_dat: list[SqwIXExperiment]

    serial_name: ClassVar[str] = "IX_experiment"
    version: ClassVar[float] = 3.0

    def _serialize_to_dict(self) -> dict[str, ir.Object]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "array_dat": ir.ObjectArray(
                ty=ir.TypeTag.struct,
                shape=(len(self.array_dat),),
                data=[exp.serialize_to_ir() for exp in self.array_dat],
            ),
        }
