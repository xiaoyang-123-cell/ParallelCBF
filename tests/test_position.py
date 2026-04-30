from __future__ import annotations

from pathlib import Path


def test_readme_contains_locked_position_paragraph() -> None:
    position = Path("docs/POSITION.md").read_text(encoding="utf-8").strip()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert position in readme
