"""Konfigurasi aplikasi: backend aktif dan path data/runs/binary core.

Semua nilai bisa ditimpa lewat environment variable. Nilai bawaan relatif
ke root repo, jadi aplikasi jalan dari clone mana pun.

    LC_BACKEND   mock | cli          (bawaan: mock)
    LC_DATA_DIR  folder dataset      (bawaan: <repo>/data)
    LC_RUNS_DIR  folder hasil run    (bawaan: <repo>/runs)
    LC_CORE_BIN  path program landcover (bawaan: <repo>/build/landcover)

Variabel lain: LC_CORE_TIMEOUT (detik, batas waktu satu run backend cli, bawaan 1800) dibaca
`launchers/cli.py`; LC_MOCK_SLEEP dibaca `launchers/mock.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BACKENDS = ("mock", "cli")


@dataclass(frozen=True)
class Config:
    backend: str
    data_dir: Path
    runs_dir: Path
    core_bin: Path


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    return Path(raw).expanduser()


def load_config() -> Config:
    """Baca konfigurasi dari environment saat ini."""
    backend = (os.getenv("LC_BACKEND") or "mock").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"LC_BACKEND={backend!r} tidak dikenal; pilih salah satu dari {', '.join(BACKENDS)}"
        )
    return Config(
        backend=backend,
        data_dir=_path_from_env("LC_DATA_DIR", ROOT / "data"),
        runs_dir=_path_from_env("LC_RUNS_DIR", ROOT / "runs"),
        core_bin=_path_from_env("LC_CORE_BIN", ROOT / "build" / "landcover"),
    )


_CONFIG = load_config()
BACKEND = _CONFIG.backend
DATA_DIR = _CONFIG.data_dir
RUNS_DIR = _CONFIG.runs_dir
CORE_BIN = _CONFIG.core_bin
