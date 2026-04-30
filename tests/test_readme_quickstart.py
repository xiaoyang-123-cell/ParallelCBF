from __future__ import annotations

import doctest
from pathlib import Path


def test_readme_quickstart_doctest() -> None:
    result = doctest.testfile(str(Path("..") / "README.md"), module_relative=True, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
