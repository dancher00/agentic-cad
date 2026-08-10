from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import NoReturn

import pytest

from da3_cad.sample import build_sample_case


@pytest.fixture(autouse=True)
def forbid_network_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CI no-network contract executable instead of documentary."""

    if os.environ.get("DA3_CAD_TEST_NO_NETWORK") != "1":
        return

    def blocked(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise RuntimeError("network access is forbidden in the CPU test suite")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def sample_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "plate"
    build_sample_case(case_dir)
    return case_dir
