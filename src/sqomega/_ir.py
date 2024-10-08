# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Intermediate representation for SQW objects."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, TypeVar

_T = TypeVar('_T')


class TypeTag(enum.Enum):
    """Single byte tag to identify types in SQW files."""

    # Gaps in values are unsupported types.
    logical = 0
    char = 1
    f64 = 3
    f32 = 4
    i8 = 5
    u8 = 6
    i32 = 9
    u32 = 10
    i64 = 11
    u64 = 12
    cell = 23
    struct = 24
    serializable = 32  # objects that 'serialize themselves'


@dataclass(kw_only=True)
class ObjectArray:
    ty: TypeTag
    shape: tuple[int, ...]
    data: list[Object]


@dataclass(kw_only=True)
class CellArray:
    shape: tuple[int, ...]
    data: list[ObjectArray]  # nested object array to encode types of each item
    ty: ClassVar[TypeTag] = TypeTag.cell


@dataclass(kw_only=True)
class Struct:
    field_names: tuple[str, ...]
    field_values: CellArray
    ty: ClassVar[TypeTag] = TypeTag.struct


@dataclass()
class String:
    value: str
    ty: ClassVar[TypeTag] = TypeTag.char


@dataclass()
class F64:
    value: float
    ty: ClassVar[TypeTag] = TypeTag.f64


@dataclass()
class U64:
    value: int
    ty: ClassVar[TypeTag] = TypeTag.u64


@dataclass()
class U32:
    value: int
    ty: ClassVar[TypeTag] = TypeTag.u32


@dataclass()
class U8:
    value: int
    ty: ClassVar[TypeTag] = TypeTag.u8


@dataclass()
class Logical:
    value: bool
    ty: ClassVar[TypeTag] = TypeTag.logical


# Not supported by SQW but represented here to simplify serialization.
@dataclass()
class Datetime:
    value: datetime
    ty: ClassVar[TypeTag] = TypeTag.char


Object = Struct | String | F64 | U64 | U32 | U8 | Logical | Datetime


class Serializable(ABC):
    @abstractmethod
    def _serialize_to_dict(self) -> dict[str, Object]: ...

    def serialize_to_ir(self) -> Struct:
        fields = self._serialize_to_dict()
        return Struct(
            field_names=tuple(fields),
            field_values=CellArray(
                shape=(len(fields), 1),  # HORACE uses a 2D array
                data=[
                    ObjectArray(ty=field.ty, shape=(1,), data=[_serialize_field(field)])
                    for field in fields.values()
                ],
            ),
        )

    def prepare_for_serialization(self: _T) -> _T:
        return self


def _serialize_field(field: Object) -> Object:
    if isinstance(field, Datetime):
        return String(value=field.value.isoformat(timespec='seconds'))
    return field
