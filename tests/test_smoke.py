"""Smoke test: the package imports and exposes a version."""

import pressflow


def test_package_imports() -> None:
    assert pressflow.__version__
