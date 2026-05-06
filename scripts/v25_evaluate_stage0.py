#!/usr/bin/env python3
"""Runtime wrapper for the V25 Stage 0 evaluator."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


DEV_ROOT = Path("/home/smartlab/parallelcbf_dev")
SITE_PACKAGES = Path("/tmp/parallelcbf_v25_day2/site")

for path in (str(DEV_ROOT), str(SITE_PACKAGES)):
    if path not in sys.path:
        sys.path.insert(0, path)

runpy.run_path(str(DEV_ROOT / "scripts" / "v25_evaluate_stage0.py"), run_name="__main__")
