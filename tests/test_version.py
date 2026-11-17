"""Check that the declared package version is consistent across the project."""

import re
from pathlib import Path

import EvoFlow


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match is not None, "version not found in pyproject.toml"
    assert match.group(1) == EvoFlow.__version__
