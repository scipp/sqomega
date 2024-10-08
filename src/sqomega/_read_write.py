# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Implementations of readers and writers for SQW object types."""

import dataclasses
from typing import Any, TypeVar

from . import _ir as ir
from ._low_level_io import LowLevelSqw

_T = TypeVar("_T")


# TODO use IR for reading
def read_objects(sqw_io: LowLevelSqw) -> Any:
    position = sqw_io.position
    type_tag = sqw_io.read_u8()
    if type_tag == ir.TypeTag.serializable.value:
        # This type object does not encode a shape, so just attempt
        # to read its contents.
        return read_objects(sqw_io)

    n_dims = sqw_io.read_u8()
    if type_tag == ir.TypeTag.char.value:
        # Special case to properly decode string
        return _squeeze(_read_char_arrays(sqw_io, n_dims))

    shape = tuple(sqw_io.read_u32() for _ in range(n_dims))

    try:
        reader = _READERS_PER_TYPE[type_tag]
    except KeyError:
        raise NotImplementedError(
            f"No reader for SQW type {type_tag} at reader position {position}"
        ) from None
    return _squeeze(_read_nd_object_array(sqw_io, shape, reader))


def write_object_array(sqw_io: LowLevelSqw, objects: ir.ObjectArray) -> None:
    position = sqw_io.position
    sqw_io.write_u8(objects.ty.value)
    sqw_io.write_u8(len(objects.shape))

    for size in objects.shape:
        sqw_io.write_u32(size)

    try:
        writer = _WRITERS_PER_TYPE[objects.ty.value]
    except KeyError:
        raise NotImplementedError(
            f"No writer for SQW type {objects.ty} at writer position {position}"
        ) from None
    for item in objects.data:
        writer(sqw_io, item)


def write_cell_array(sqw_io: LowLevelSqw, cells: ir.CellArray) -> None:
    sqw_io.write_u8(cells.ty.value)
    sqw_io.write_u8(len(cells.shape))
    for size in cells.shape:
        sqw_io.write_u32(size)
    for item in cells.data:
        write_object_array(sqw_io, item)


# TODO use CellArray
def _read_nd_object_array(
    sqw_io: LowLevelSqw, shape: tuple[int, ...], reader
) -> list[Any]:
    if len(shape) == 1:
        return [reader(sqw_io) for _ in range(shape[0])]
    return [_read_nd_object_array(sqw_io, shape[1:], reader) for _ in range(shape[0])]


def _read_char_arrays(sqw_io: LowLevelSqw, n_dims: int) -> str:
    if n_dims == 0:
        return ""
    if n_dims != 1:
        raise NotImplementedError(
            f"Cannot read char arrays with more than one dimension, got {n_dims}"
        )
    return sqw_io.read_char_array()


def _write_char_array(sqw_io: LowLevelSqw, value: ir.Object) -> None:
    chars: ir.String = value
    sqw_io.write_chars(chars.value)


def _read_cell(sqw_io: LowLevelSqw) -> Any:
    return read_objects(sqw_io)


def _read_struct(sqw_io: LowLevelSqw) -> Any:
    n_fields = sqw_io.read_u32()
    field_name_sizes = [sqw_io.read_u32() for _ in range(n_fields)]
    field_names = [sqw_io.read_n_chars(size) for size in field_name_sizes]
    field_values = read_objects(sqw_io)
    field_values = _squeeze(field_values)

    # TODO IR instead of dataclass
    #   not sure about squeezing
    return dataclasses.make_dataclass(
        _struct_serial_name(field_names, field_values),
        (
            _make_struct_field(name, value)
            for name, value in zip(field_names, field_values, strict=True)
        ),
        frozen=True,
        slots=True,
    )()


def _make_struct_field(name: str, value: Any) -> tuple[str, Any, dataclasses.Field]:
    return name, type(value), dataclasses.field(default_factory=lambda: value)


def _struct_serial_name(field_names: list[str], field_values: list[Any]) -> str:
    try:
        return str(
            next(
                iter(
                    value
                    for name, value in zip(field_names, field_values, strict=True)
                    if name == 'serial_name'
                )
            )
        )
    except StopIteration:
        return "unknown"


def _write_struct(sqw_io: LowLevelSqw, obj: ir.Object) -> None:
    struct: ir.Struct = obj
    sqw_io.write_u32(len(struct.field_names))
    for name in struct.field_names:
        sqw_io.write_u32(len(name))
    for name in struct.field_names:
        sqw_io.write_chars(name)
    write_cell_array(sqw_io, struct.field_values)


def _write_f64(sqw_io: LowLevelSqw, value: ir.Object) -> None:
    f64: ir.F64 = value
    sqw_io.write_f64(f64.value)


def _write_logical(sqw_io: LowLevelSqw, value: ir.Object) -> None:
    logical: ir.Logical = value
    sqw_io.write_logical(logical.value)


_READERS_PER_TYPE = {
    ir.TypeTag.logical.value: LowLevelSqw.read_logical,
    ir.TypeTag.f64.value: LowLevelSqw.read_f64,
    ir.TypeTag.cell.value: _read_cell,
    ir.TypeTag.struct.value: _read_struct,
}

_WRITERS_PER_TYPE = {
    ir.TypeTag.logical.value: _write_logical,
    ir.TypeTag.char.value: _write_char_array,
    ir.TypeTag.f64.value: _write_f64,
    ir.TypeTag.struct.value: _write_struct,
}


def _squeeze(nd_list: list[_T] | _T) -> list[_T] | _T:
    if isinstance(nd_list, list):
        if len(nd_list) == 1:
            return _squeeze(nd_list[0])
        return [_squeeze(item) for item in nd_list]
    return nd_list
