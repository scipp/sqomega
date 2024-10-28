# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import Any

import pytest

from sqomega import Sqw

pace_neutrons = pytest.importorskip("pace_neutrons")


@pytest.fixture(scope='module')
def matlab() -> Any:
    return pace_neutrons.Matlab()


def test_load_empty_file(matlab: Any) -> None:
    with Sqw.build("empty.sqw").create():
        pass

    obj = matlab.read_horace("empty.sqw")
    print(obj)  # noqa: T201
    pytest.fail("Not implemented")
