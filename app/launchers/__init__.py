"""Launcher: pembuat folder run (MockLauncher untuk data tiruan, CliLauncher untuk program landcover)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def make_launcher(
    backend: str,
    data_dir: Path | str,
    core_bin: "str | os.PathLike[str] | Sequence[str] | None" = None,
):
    """Launcher untuk `backend` (nilai `config.BACKEND`).

    Tempat satu-satunya yang memilih launcher. `core_bin` (`config.CORE_BIN`) hanya dipakai backend
    `cli`: path program `landcover`, atau awalan perintah (mis. pembungkus ke WSL)."""
    if backend == "mock":
        from launchers.mock import MockLauncher

        return MockLauncher(data_dir)
    if backend == "cli":
        from launchers.cli import CliLauncher

        if core_bin is None:
            raise ValueError("backend cli butuh core_bin (path program landcover; LC_CORE_BIN)")
        return CliLauncher(data_dir, core_bin)
    raise ValueError(f"backend tidak dikenal: {backend!r}")
