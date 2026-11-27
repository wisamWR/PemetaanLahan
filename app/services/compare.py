"""Perbandingan CPU saja vs Optimalisasi (logika murni, tanpa Streamlit).

Diekstrak dari `ui/compare.py`. `speedup`, `format_seconds`, `comparable_runs`, `timer_text`, dan
`device_caption` dipakai bersama oleh Streamlit dan API.
"""

from __future__ import annotations

from typing import Mapping

from landcover.contract import RunResult
from services.runs import DEVICE_LABELS

MOCK_NOTICE = "DATA TIRUAN: bukan hasil pengukuran"
EMPTY = "—"


def format_seconds(seconds: float | None) -> str:
    """Detik dengan dua desimal ("12.34 s"); `EMPTY` bila tidak ada."""
    return EMPTY if seconds is None else f"{seconds:.2f} s"


def speedup(t_cpu: float | None, t_opt: float | None) -> float | None:
    """T_cpu / T_opt; None bila salah satu belum ada atau T_opt tidak positif."""
    if t_cpu is None or t_opt is None or not t_opt > 0:
        return None
    return t_cpu / t_opt


def pair_key(run: RunResult) -> tuple:
    m = run.meta
    return (m.get("dataset"), run.k, m.get("iters"), m.get("seed"))


def comparable_runs(
    runs: Mapping[str, RunResult], active_id: str
) -> tuple[RunResult | None, RunResult | None]:
    """(run CPU saja, run Optimalisasi) yang sebanding dengan run aktif.

    Sebanding = dataset, K, iterasi, dan seed sama. Run aktif dipakai untuk sisinya; sisi
    lain adalah yang TERBARU di `runs` (urutan penyisipan; yang baru dijalankan ada di
    akhir). Sisi yang belum dijalankan = None."""
    active = runs.get(active_id)
    if active is None:
        return None, None
    key = pair_key(active)
    cpu = opt = None
    for run in runs.values():
        if pair_key(run) != key:
            continue
        if run.mode == "cpu":
            cpu = run
        else:
            opt = run
    if active.mode == "cpu":
        cpu = active
    else:
        opt = active
    return cpu, opt


def timer_text(label: str, seconds: float) -> str:
    return f"⏱ **{label}** berjalan: {seconds:.1f} s"


def device_caption(run: RunResult) -> str:
    device = "serial" if run.mode == "cpu" else run.kmeans_device
    text = f"K-Means: {DEVICE_LABELS.get(device, device)}"
    if device == "omp":            # jumlah thread hanya bermakna untuk OpenMP (kontrak U6: cuda = 0)
        text += f", {run.meta.get('threads', '?')} thread"
    return text


__all__ = [
    "EMPTY", "MOCK_NOTICE", "comparable_runs", "device_caption", "format_seconds", "pair_key",
    "speedup", "timer_text",
]
