# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Implementations of readers and writers for SQW object types."""

import dataclasses
from typing import Any, TypeVar

from ._bytes import TypeTag
from ._low_level_io import LowLevelSqw

_T = TypeVar("_T")


def read_objects(sqw_io: LowLevelSqw) -> Any:
    position = sqw_io.position
    type_tag = sqw_io.read_u8()
    if type_tag == TypeTag.serializable.value:
        # This type object does not encode a shape, so just attempt
        # to read its contents.
        return read_objects(sqw_io)

    n_dims = sqw_io.read_u8()
    if type_tag == TypeTag.char.value:
        # Special case to properly decode string
        return _squeeze(_read_char_arrays(sqw_io, n_dims))

    shape = tuple(sqw_io.read_u32() for _ in range(n_dims))

    try:
        reader = _READERS_PER_TYPE[type_tag]
    except KeyError:
        raise NotImplementedError(
            f"No reader for SQW type {type_tag} at position {position}"
        ) from None
    return _squeeze(_read_nd_object_array(sqw_io, shape, reader))


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


def _read_cell(sqw_io: LowLevelSqw) -> Any:
    return read_objects(sqw_io)


def _read_struct(sqw_io: LowLevelSqw) -> Any:
    n_fields = sqw_io.read_u32()
    field_name_sizes = [sqw_io.read_u32() for _ in range(n_fields)]
    field_names = [sqw_io.read_n_chars(size) for size in field_name_sizes]
    field_values = read_objects(sqw_io)
    field_values = _squeeze(field_values)

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


_READERS_PER_TYPE = {
    TypeTag.logical.value: LowLevelSqw.read_logical,
    TypeTag.f64.value: LowLevelSqw.read_f64,
    TypeTag.cell.value: _read_cell,
    TypeTag.struct.value: _read_struct,
}


def _squeeze(nd_list: list[_T] | _T) -> list[_T] | _T:
    if isinstance(nd_list, list):
        if len(nd_list) == 1:
            return _squeeze(nd_list[0])
        return [_squeeze(item) for item in nd_list]
    return nd_list
