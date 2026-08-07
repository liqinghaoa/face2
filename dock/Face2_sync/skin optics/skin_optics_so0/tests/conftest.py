"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from skin_optics.assets import load_assets


@pytest.fixture(scope="session")
def assets5():
    return load_assets(5)


@pytest.fixture(scope="session")
def assets1():
    return load_assets(1)
