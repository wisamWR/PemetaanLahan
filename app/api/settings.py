"""Pengaturan API: backend, folder data/runs, program core, dan folder hasil build web.

`Settings.from_config()` membaca atribut modul `config` pada saat dipanggil (bukan saat impor), jadi
tes dan proses yang mengubah `config` mendapat nilai terbaru. Tes membuat `Settings(...)` langsung.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import config


@dataclass(frozen=True)
class Settings:
    backend: str                                           # "mock" | "cli"
    data_dir: Path
    runs_dir: Path
    core_bin: "str | os.PathLike[str] | Sequence[str]"     # path atau awalan perintah (pembungkus WSL)
    web_dist: Path | None = None                           # web/dist; None = tidak menyajikan frontend

    def __post_init__(self) -> None:
        if self.backend not in config.BACKENDS:
            raise ValueError(f"backend {self.backend!r} tidak dikenal; pilih salah satu dari {', '.join(config.BACKENDS)}")

    @classmethod
    def from_config(cls, web_dist: Path | str | None = None) -> "Settings":
        dist = Path(web_dist) if web_dist is not None else config.ROOT / "web" / "dist"
        return cls(
            backend=config.BACKEND,
            data_dir=Path(config.DATA_DIR),
            runs_dir=Path(config.RUNS_DIR),
            core_bin=config.CORE_BIN,
            web_dist=dist,
        )

    @property
    def mock(self) -> bool:
        return self.backend == "mock"
