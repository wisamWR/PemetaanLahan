"""Kartu perbandingan CPU saja vs Optimalisasi dan timer saat run berjalan (Langkah 6, butir 3).

`speedup`, `format_seconds`, `comparable_runs`, dan `timer_text` murni. `render_cards` dan
`render_timer` hanya menggambar.
"""

from __future__ import annotations

from typing import Mapping

import streamlit as st

from landcover.contract import RunResult
from ui.badges import MOCK_BADGE
from ui.runs import DEVICE_LABELS, MODE_LABELS

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


def _key(run: RunResult) -> tuple:
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
    key = _key(active)
    cpu = opt = None
    for run in runs.values():
        if _key(run) != key:
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


def render_timer(placeholder, label: str, seconds: float) -> None:
    """Perbarui timer di `placeholder` (st.empty). Dipanggil dari thread skrip, bukan pekerja."""
    placeholder.markdown(timer_text(label, seconds))


def _device_caption(run: RunResult) -> str:
    device = "serial" if run.mode == "cpu" else run.kmeans_device
    text = f"K-Means: {DEVICE_LABELS.get(device, device)}"
    if device == "omp":            # jumlah thread hanya bermakna untuk OpenMP (kontrak U6: cuda = 0)
        text += f", {run.meta.get('threads', '?')} thread"
    return text


def render_cards(
    cpu: RunResult | None,
    opt: RunResult | None,
    notes: Mapping[str, str] | None = None,
    *,
    mock: bool = False,
) -> None:
    """Tiga `st.metric`: total CPU saja, total Optimalisasi, dan speedup.

    `notes`: run_id -> catatan (mis. "hasil tersimpan, 21-11-2026 10:15"), ditampilkan di
    bawah kartu run itu. Waktu total dari `timing.json` (bukan jumlah tahap)."""
    notes = notes or {}
    s = speedup(cpu.total if cpu else None, opt.total if opt else None)
    c_cpu, c_opt, c_speed = st.columns(3)
    for col, title, run, empty_hint in (
        (c_cpu, MODE_LABELS["cpu"], cpu, "belum dijalankan (tekan Jalankan dengan mode CPU saja)"),
        (c_opt, MODE_LABELS["opt"], opt, "belum dijalankan (tekan Jalankan dengan mode Optimalisasi)"),
    ):
        with col:
            st.metric(f"Total: {title}", format_seconds(run.total if run else None))
            if run is None:
                st.caption(empty_hint)
            else:
                st.caption(_device_caption(run))
                note = notes.get(run.path.name)
                if note:
                    st.caption(note)
    with c_speed:
        st.metric("Speedup (T_cpu / T_opt)", EMPTY if s is None else f"{s:.2f}×")
        if s is None:
            st.caption("butuh kedua run dengan dataset, K, iterasi, dan seed yang sama")
    if mock:
        st.caption(f"{MOCK_BADGE} bukan hasil pengukuran")


__all__ = [
    "EMPTY", "MOCK_NOTICE", "comparable_runs", "format_seconds", "render_cards", "render_timer",
    "speedup", "timer_text",
]
