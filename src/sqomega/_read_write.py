# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Implementations of readers and writers for SQW object types."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from . import _ir as ir
from ._low_level_io import LowLevelSqw

_Shape = tuple[int, ...]
_T = TypeVar("_T")
_AnyObjectList = list[ir.Object] | list[ir.ObjectArray] | npt.NDArray[Any]
_ObjectReader = Callable[[LowLevelSqw, _Shape], _AnyObjectList]
_ObjectWriter = Callable[[LowLevelSqw, _AnyObjectList], None]


class _IORegistry(Generic[_T]):
    def __init__(self, action: str) -> None:
        self._action = action
        self._registry: dict[ir.TypeTag, _T] = {}

    def add(self, ty: ir.TypeTag) -> Callable[[_T], _T]:
        def add_impl(impl: _T) -> _T:
            if ty in self._registry:
                raise ValueError(f"Duplicate registration for {self._action} type {ty}")
            self._registry[ty] = impl
            return impl

        return add_impl

    def get(self, ty: ir.TypeTag, pos: int) -> _T:
        try:
            return self._registry[ty]
        except KeyError:
            raise ValueError(
                f"No {self._action} for SQW type {ty} as position {pos}"
            ) from None


_READERS = _IORegistry[_ObjectReader]("reader")
_WRITERS = _IORegistry[_ObjectWriter]("writer")


def read_object_array(sqw_io: LowLevelSqw) -> ir.ObjectArray | ir.CellArray:
    position = sqw_io.position
    ty = ir.TypeTag(sqw_io.read_u8())
    if ty == ir.TypeTag.serializable:  # TODO
        # This type object does not encode a shape, so just attempt
        # to read its contents.
        return read_object_array(sqw_io)

    n_dims = sqw_io.read_u8()
    shape = tuple(sqw_io.read_u32() for _ in range(n_dims))

    reader = _READERS.get(ty, position)
    data = reader(sqw_io, shape)
    if ty == ir.TypeTag.cell:
        return ir.CellArray(shape=shape, data=data)  # type: ignore[arg-type]
    return ir.ObjectArray(ty=ty, shape=shape, data=data)  # type: ignore[arg-type]


def write_object_array(
    sqw_io: LowLevelSqw, objects: ir.ObjectArray | ir.CellArray
) -> None:
    position = sqw_io.position
    sqw_io.write_u8(objects.ty.value)
    sqw_io.write_u8(len(objects.shape))
    for size in objects.shape:
        sqw_io.write_u32(size)

    writer = _WRITERS.get(objects.ty, position)
    writer(sqw_io, objects.data)


@_READERS.add(ir.TypeTag.char)
def _read_char_arrays(sqw_io: LowLevelSqw, shape: _Shape) -> list[ir.Object]:
    # TODO is the str length shape[0] or shape[-1]?
    if not shape:
        return [ir.String("")]
    return [ir.String(sqw_io.read_n_chars(shape[0])) for _ in range(_volume(shape[1:]))]


@_WRITERS.add(ir.TypeTag.char)
def _write_char_array(sqw_io: LowLevelSqw, objects: _AnyObjectList) -> None:
    for obj in objects:
        chars: ir.String = obj  # type: ignore[assignment]
        sqw_io.write_chars(chars.value)


@_READERS.add(ir.TypeTag.cell)
def _read_cell(sqw_io: LowLevelSqw, shape: _Shape) -> Any:
    return [read_object_array(sqw_io) for _ in range(_volume(shape))]


@_WRITERS.add(ir.TypeTag.cell)
def _write_cell(sqw_io: LowLevelSqw, objects: _AnyObjectList) -> None:
    obj_arrays: list[ir.ObjectArray] = objects  # type: ignore[assignment]
    for obj in obj_arrays:
        write_object_array(sqw_io, obj)


@_READERS.add(ir.TypeTag.struct)
def _read_struct(sqw_io: LowLevelSqw, shape: _Shape) -> list[ir.Object]:
    return [_read_single_struct(sqw_io) for _ in range(_volume(shape))]


def _read_single_struct(sqw_io: LowLevelSqw) -> ir.Struct:
    n_fields = sqw_io.read_u32()
    field_name_sizes = [sqw_io.read_u32() for _ in range(n_fields)]
    field_names = tuple(sqw_io.read_n_chars(size) for size in field_name_sizes)
    field_values: ir.CellArray = _expect_ty(ir.TypeTag.cell, read_object_array(sqw_io))  # type: ignore[assignment]
    return ir.Struct(field_names=field_names, field_values=field_values)


@_WRITERS.add(ir.TypeTag.struct)
def _write_struct(sqw_io: LowLevelSqw, objects: _AnyObjectList) -> None:
    for obj in objects:
        struct: ir.Struct = obj  # type: ignore[assignment]
        _write_single_struct(sqw_io, struct)


def _write_single_struct(sqw_io: LowLevelSqw, struct: ir.Struct) -> None:
    sqw_io.write_u32(len(struct.field_names))
    for name in struct.field_names:
        sqw_io.write_u32(len(name))
    for name in struct.field_names:
        sqw_io.write_chars(name)
    write_object_array(sqw_io, struct.field_values)


@_READERS.add(ir.TypeTag.f64)
def _read_f64(
    sqw_io: LowLevelSqw, shape: _Shape
) -> list[ir.Object] | npt.NDArray[np.float64]:
    data = sqw_io.read_array(shape, np.dtype('float64'))
    if data.size == 1:
        return [ir.F64(data.squeeze().item())]
    return data


@_WRITERS.add(ir.TypeTag.f64)
def _write_f64(sqw_io: LowLevelSqw, objects: _AnyObjectList) -> None:
    if isinstance(objects, np.ndarray):
        sqw_io.write_array(objects)
    else:
        for obj in objects:
            f64: ir.F64 = obj  # type: ignore[assignment]
            sqw_io.write_f64(f64.value)


@_READERS.add(ir.TypeTag.logical)
def _read_logical(sqw_io: LowLevelSqw, shape: _Shape) -> list[ir.Object]:
    return [ir.Logical(sqw_io.read_logical()) for _ in range(_volume(shape))]


@_WRITERS.add(ir.TypeTag.logical)
def _write_logical(sqw_io: LowLevelSqw, objects: _AnyObjectList) -> None:
    for obj in objects:
        logical: ir.Logical = obj  # type: ignore[assignment]
        sqw_io.write_logical(logical.value)


_O = TypeVar("_O", bound=ir.Object | ir.ObjectArray | ir.CellArray)


def _expect_ty(ty: ir.TypeTag, obj: _O) -> _O:
    if obj.ty != ty:
        raise TypeError(f"Expected {ty}, got {obj.ty}")
    return obj


def _volume(shape: _Shape) -> int:
    return int(np.prod(shape))
