"""Kartu perbandingan CPU saja vs Optimalisasi dan timer saat run berjalan (Langkah 6, butir 3).

`speedup`, `format_seconds`, `comparable_runs`, dan `timer_text` murni dan ada di
`services/compare.py` (dipakai bersama API FastAPI); modul ini mengimpornya, jadi nama-nama lama
tetap berlaku. `render_cards` dan `render_timer` hanya menggambar.
"""

from __future__ import annotations

from typing import Mapping

import streamlit as st

from landcover.contract import RunResult
from services.compare import (
    EMPTY,
    MOCK_NOTICE,
    comparable_runs,
    device_caption,
    format_seconds,
    speedup,
    timer_text,
)
from ui.badges import MOCK_BADGE
from ui.runs import MODE_LABELS

_device_caption = device_caption        # nama lama (dipakai tes)


def render_timer(placeholder, label: str, seconds: float) -> None:
    """Perbarui timer di `placeholder` (st.empty). Dipanggil dari thread skrip, bukan pekerja."""
    placeholder.markdown(timer_text(label, seconds))


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
                st.caption(device_caption(run))
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
