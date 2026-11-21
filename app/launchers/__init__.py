"""Launcher: pembuat folder run (MockLauncher sekarang, CliLauncher di Langkah 11)."""

from __future__ import annotations

from pathlib import Path


def make_launcher(backend: str, data_dir: Path | str):
    """Launcher untuk `backend` (nilai `config.BACKEND`).

    Tempat satu-satunya yang memilih launcher; Langkah 11 menambah cabang `cli`."""
    if backend == "mock":
        from launchers.mock import MockLauncher

        return MockLauncher(data_dir)
    if backend == "cli":
        raise NotImplementedError(
            "backend cli belum tersedia: CliLauncher dibuat di Langkah 11 "
            "(butuh program landcover, Langkah 11a). Pakai LC_BACKEND=mock untuk sekarang."
        )
    raise ValueError(f"backend tidak dikenal: {backend!r}")
