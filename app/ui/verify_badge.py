"""Lencana status verifikasi antarmode (plan Langkah 9; logika di `landcover/verify.py`).

Kalimat vonis, sumber, dan rincian (`verdict_text`, `source_text`, `detail_rows`, `format_*`) murni
dan ada di `services/verification.py` (dipakai juga oleh API FastAPI); modul ini mengimpornya, jadi
nama-nama lama tetap berlaku. Di sini tinggal yang khusus Streamlit: `escape_markdown`,
`verdict_style` (tingkat vonis -> warna teks Streamlit), cache, dan `render_*`.

Aturan kata (jangan dilonggarkan) ada di docstring `services/verification.py`. Vonis "Berbeda"
bukan galat aplikasi: ditampilkan sebagai teks berwarna dengan ikon dan kata, bukan `st.error`.
"""

from __future__ import annotations

import re

import streamlit as st

from landcover.contract import RunResult
from landcover.verify import Verification, verification_stamp, verify_pair
from services.verification import (
    ICON_BAD,
    ICON_OK,
    ICON_PARTIAL,
    ICON_WARN,
    LEVEL_BAD,
    LEVEL_ICONS,
    LEVEL_OK,
    LEVEL_PARTIAL,
    LEVEL_WARN,
    MOCK_VERIFY_NOTICE,
    TITLE,
    detail_rows,
    format_identical_pct,
    format_pixels,
    params_text as _params_text,
    source_text,
    verdict_level,
    verdict_text,
)
from ui.badges import MOCK_BADGE
from ui.charts import run_label
from ui.runs import MODE_LABELS

_LEVEL_COLORS = {LEVEL_OK: "green", LEVEL_PARTIAL: "blue", LEVEL_BAD: "red", LEVEL_WARN: "orange"}

_MD_SPECIAL = re.compile(r"([\\`*_\[\]$~])")


def escape_markdown(text: str) -> str:
    """Hindari nama dataset atau karakter lain mengubah markdown (miring, tautan, LaTeX)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def verdict_style(v: Verification) -> tuple[str, str]:
    """(ikon, warna teks Streamlit) untuk vonis: hijau hanya untuk identik di resolusi penuh."""
    level = verdict_level(v)
    return LEVEL_ICONS[level], _LEVEL_COLORS[level]



# --- cache ---------------------------------------------------------------------- #

@st.cache_data(show_spinner="Membandingkan hasil CPU saja dan Optimalisasi…", max_entries=32)
def _verify_cached(_ref: RunResult, _run: RunResult, ref_stamp: tuple, run_stamp: tuple) -> Verification:  # noqa: ARG001
    return verify_pair(_ref, _run)


def verify_cached(ref: RunResult, run: RunResult) -> Verification:
    """`verify_pair` dengan cache berkunci (path, waktu ubah, ukuran) berkas yang dibaca.

    Halaman dijalankan ulang pada setiap suntingan panel pelabelan; tanpa cache, membaca dua
    `labels.u8` 120 MB di tiap suntingan akan merusak respons panel (Langkah 8: < 1 detik)."""
    return _verify_cached(ref, run, verification_stamp(ref), verification_stamp(run))

# --- tampilan ------------------------------------------------------------------- #

def render_verification(cpu: RunResult, opt: RunResult) -> None:
    """Vonis, sumber, catatan, dan rincian untuk pasangan (CPU saja, Optimalisasi)."""
    try:
        v = verify_cached(cpu, opt)
    except (OSError, ValueError) as exc:     # batas UI: verifikasi gagal bukan alasan halaman mogok
        st.warning(f"Verifikasi tidak bisa dijalankan: {escape_markdown(str(exc))}")
        return
    icon, color = verdict_style(v)
    st.markdown(f":{color}[**{icon} {escape_markdown(verdict_text(v))}**]")
    st.caption(f"Pembanding: {run_label(cpu)} ↔ {run_label(opt)}")
    source = source_text(v)
    if source:
        st.caption(source)
    for note in v.notes:
        st.caption(f"Catatan: {escape_markdown(note)}")      # memuat teks dari berkas run
    if v.mock:
        st.caption(MOCK_VERIFY_NOTICE)
    rows = detail_rows(v)
    if rows:
        with st.expander("Rincian verifikasi"):
            st.markdown("\n".join(f"- **{what}**: {escape_markdown(result)}" for what, result in rows))


def render_verify_slot(cpu: RunResult | None, opt: RunResult | None) -> None:
    """Kotak verifikasi di halaman. `cpu`/`opt` = pasangan sebanding dari `comparable_runs`.

    Tanpa run: kotak kosong dengan petunjuk. Satu sisi saja: petunjuk run yang kurang (ini bukan
    "tidak sebanding": run yang tidak sebanding tidak pernah dipasangkan)."""
    with st.container(border=True):
        if cpu is None and opt is None:
            st.caption(f"{TITLE}: belum ada hasil (jalankan dulu)")
            return
        mock = any(r.mock for r in (cpu, opt) if r is not None)
        st.subheader(f"{TITLE} {MOCK_BADGE}" if mock else TITLE)
        if cpu is None or opt is None:
            have = cpu or opt
            need = MODE_LABELS["opt" if opt is None else "cpu"]
            st.caption(
                f"Butuh kedua run untuk membandingkan. Jalankan {need} untuk {escape_markdown(_params_text(have))} "
                "(dataset, K, iterasi, dan seed harus sama)."
            )
            return
        render_verification(cpu, opt)


__all__ = [
    "ICON_BAD", "ICON_OK", "ICON_PARTIAL", "ICON_WARN", "MOCK_VERIFY_NOTICE", "TITLE",
    "detail_rows", "escape_markdown", "format_identical_pct", "format_pixels", "render_verification",
    "render_verify_slot", "source_text", "verdict_style", "verdict_text", "verify_cached",
]
