"""Atomic checkpoint writer."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any


class AtomicCheckpoint:
    """Persist picklable payloads using `.tmp -> fsync -> rename`."""

    def save(self, payload: object, path: str | Path) -> Path:
        """Atomically write a checkpoint payload."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.name}.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        self._fsync_parent(target.parent)
        return target

    def load(self, path: str | Path) -> Any:
        """Load a checkpoint payload."""

        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _fsync_parent(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
