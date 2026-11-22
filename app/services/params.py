"""Isian form parameter run dan pembuatan `RunParams` (logika murni, tanpa Streamlit).

Diekstrak dari `ui/sidebar.py`. Dipakai bersama oleh sidebar Streamlit dan API (`POST /api/runs`).
"""

from __future__ import annotations

from dataclasses import dataclass

from launchers.base import RunParams

K_MIN, K_DEFAULT = 2, 8
ITERS_DEFAULT, ITERS_MAX = 20, 1000
SEED_DEFAULT = 42
THREADS_MIN, THREADS_MAX, THREADS_DEFAULT = 1, 20, 8


@dataclass(frozen=True)
class FormValues:
    dataset: str
    k: int = K_DEFAULT
    iters: int = ITERS_DEFAULT
    mode: str = "cpu"                  # "cpu" | "opt"
    kmeans: str = "auto"               # "auto" | "omp" | "cuda"
    threads: int = THREADS_DEFAULT
    seed: int = SEED_DEFAULT
    force: bool = False                # paksa jalankan ulang (abaikan hasil CPU tersimpan)


def build_params(values: FormValues, mode: str | None = None) -> RunParams:
    """`RunParams` dari isian form; `mode` menimpa pilihan mode (untuk Bandingkan).

    ValueError dari `RunParams` (nama dataset tidak aman, K di luar rentang, dst.)
    dibiarkan naik: pemanggil menampilkannya sebagai pesan."""
    return RunParams(
        dataset=values.dataset,
        mode=values.mode if mode is None else mode,
        kmeans=values.kmeans,
        k=values.k,
        iters=values.iters,
        seed=values.seed,
        threads=values.threads,
    )


def build_pair(values: FormValues) -> tuple[RunParams, RunParams]:
    """(CPU saja, Optimalisasi) dengan parameter lain yang sama, untuk tombol Bandingkan."""
    return build_params(values, "cpu"), build_params(values, "opt")


__all__ = [
    "ITERS_DEFAULT", "ITERS_MAX", "K_DEFAULT", "K_MIN", "SEED_DEFAULT", "THREADS_DEFAULT",
    "THREADS_MAX", "THREADS_MIN", "FormValues", "build_pair", "build_params",
]
