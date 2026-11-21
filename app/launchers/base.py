"""Parameter run, run_id, dan protokol Launcher (plan Langkah 3 dan Bagian 3).

UI hanya membaca folder run. Yang berbeda antar backend cuma launcher yang
menulisnya: MockLauncher (data tiruan, bertanda "mock") atau CliLauncher
(subprocess ke program `landcover`, Langkah 11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from landcover.reference_kmeans import KMEANS_MAX_K, MAX_SEED

MODES = ("cpu", "opt")
KMEANS_CHOICES = ("auto", "omp", "cuda")

# LC_BACKEND (config.BACKENDS) -> subfolder di runs/ (plan 3: runs/mock/ dan runs/core/).
BACKEND_DIRS = {"mock": "mock", "cli": "core"}

# Nama dataset dipakai sebagai bagian nama folder: huruf/angka/_ . - spasi di tengah,
# tanpa pemisah path dan tanpa ":" (aman di Windows dan WSL).
_SAFE_NAME = re.compile(r"^\w(?:[\w .-]*\w)?$")

ProgressCallback = Callable[[str], None]


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class RunParams:
    dataset: str
    mode: str          # "cpu" | "opt"
    kmeans: str        # "auto" | "omp" | "cuda" (diabaikan bila mode == "cpu")
    k: int
    iters: int
    seed: int
    threads: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not _SAFE_NAME.match(self.dataset):
            raise ValueError(
                f"nama dataset tidak aman atau kosong: {self.dataset!r} "
                "(boleh huruf, angka, _ . - dan spasi di tengah; tanpa pemisah path)"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode harus salah satu dari {MODES}, ditemukan {self.mode!r}")
        if self.kmeans not in KMEANS_CHOICES:
            raise ValueError(
                f"kmeans harus salah satu dari {KMEANS_CHOICES}, ditemukan {self.kmeans!r}"
            )
        for name in ("k", "iters", "seed", "threads"):
            if not _is_int(getattr(self, name)):
                raise ValueError(f"{name} harus bilangan bulat, ditemukan {getattr(self, name)!r}")
        if not 1 <= self.k <= KMEANS_MAX_K:
            raise ValueError(f"k harus 1..{KMEANS_MAX_K}, ditemukan {self.k}")
        if self.iters < 1:
            raise ValueError(f"iters harus >= 1, ditemukan {self.iters}")
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed harus 0..{MAX_SEED} (uint32), ditemukan {self.seed}")
        if self.threads < 1:
            raise ValueError(f"threads harus >= 1, ditemukan {self.threads}")


def run_id(p: RunParams) -> str:
    """Nama folder run, mis. "sleman__opt-auto__k8_it20_s42_t8".

    Mode cpu dinormalkan ("cpu", t1): pilihan kmeans dan jumlah thread tidak
    memengaruhi hasilnya, jadi tidak boleh membuat folder terpisah."""
    if p.mode == "cpu":
        engine, threads = "cpu", 1
    else:
        engine, threads = f"opt-{p.kmeans}", p.threads
    return f"{p.dataset}__{engine}__k{p.k}_it{p.iters}_s{p.seed}_t{threads}"


def out_dir_for(runs_dir: Path | str, backend: str, p: RunParams) -> Path:
    """Folder run untuk `p`: <runs_dir>/mock/<run_id> atau <runs_dir>/core/<run_id>.

    Hasil tiruan dan hasil core sengaja dipisah folder (plan Bagian 3, aturan 5)."""
    try:
        sub = BACKEND_DIRS[backend]
    except KeyError:
        raise ValueError(
            f"backend tidak dikenal: {backend!r} (yang dikenal: {', '.join(BACKEND_DIRS)})"
        ) from None
    return Path(runs_dir) / sub / run_id(p)


class Launcher(Protocol):
    def run(
        self,
        params: RunParams,
        out_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Tulis folder run (kontrak 4.3) ke `out_dir` (folder run itu sendiri, seperti
        `--out` pada program inti) lalu kembalikan path-nya."""
        ...


__all__ = [
    "BACKEND_DIRS", "KMEANS_CHOICES", "MODES", "Launcher", "ProgressCallback",
    "RunParams", "out_dir_for", "run_id",
]
