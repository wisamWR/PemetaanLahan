"""Lencana status verifikasi antarmode (plan Langkah 9; logika di `landcover/verify.py`).

Fungsi teks (`verdict_text`, `verdict_style`, `source_text`, `detail_rows`, `format_*`) murni dan
diuji tanpa browser. `render_verify_slot` hanya menggambar.

Aturan kata (jangan dilonggarkan):
  * "Label identik 100 % (N piksel), centroid sama" HANYA untuk sumber yang menjangkau label resolusi
    penuh (`labels.u8` atau `verification.json`). Hasil dari pratinjau selalu berkata "Verifikasi
    parsial" dan hanya menyebut jumlah piksel pratinjau.
  * Persen identik dipotong ke bawah (bukan dibulatkan): satu piksel berbeda dari 120 juta tidak
    boleh tampil "100.00 %".
  * Vonis "Berbeda" bukan galat aplikasi: ditampilkan sebagai teks berwarna dengan ikon dan kata,
    bukan `st.error` (yang dipakai untuk kegagalan aplikasi).
  * Format angka mengikuti aplikasi (ribuan koma, desimal titik, mis. 120,560,400 dan 99.98 %),
    bukan tanda baca Indonesia yang dicontohkan di plan.
"""

from __future__ import annotations

import re
from decimal import ROUND_DOWN, Decimal

import streamlit as st

from landcover.contract import RunResult
from landcover.verify import (
    DIFF_CENTROIDS,
    DIFF_COUNTS,
    DIFF_LABELS,
    SOURCE_FILE,
    SOURCE_LABELS,
    SOURCE_PREVIEW,
    STATUS_DIFFERENT,
    STATUS_IDENTICAL,
    STATUS_NOT_COMPARABLE,
    VERIFICATION_FILE,
    Verification,
    verification_stamp,
    verify_pair,
)
from ui.badges import MOCK_BADGE
from ui.charts import run_label
from ui.runs import MODE_LABELS

TITLE = "Status verifikasi antarmode"
MOCK_VERIFY_NOTICE = (
    "DATA TIRUAN: hasil verifikasi ini tidak membuktikan apa pun tentang core C++/OpenMP/CUDA."
)

ICON_OK, ICON_PARTIAL, ICON_BAD, ICON_WARN = "✅", "ℹ️", "❌", "⚠️"


# --- pemformatan ---------------------------------------------------------------- #

def format_pixels(n: int) -> str:
    """Jumlah piksel dengan pemisah ribuan koma (mis. 120,560,400)."""
    return f"{int(n):,}"


def format_identical_pct(pct: float) -> str:
    """Persen identik DIPOTONG ke bawah dengan dua desimal ("99.98"); tepat 100 jadi "100".

    Memotong menjamin nilai di bawah 100 tidak pernah tampil sebagai "100.00"."""
    if pct >= 100:
        return "100"
    return f"{Decimal(repr(float(pct))).quantize(Decimal('0.01'), rounding=ROUND_DOWN):f}"


_MD_SPECIAL = re.compile(r"([\\`*_\[\]$~])")


def escape_markdown(text: str) -> str:
    """Hindari nama dataset atau karakter lain mengubah markdown (miring, tautan, LaTeX)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


# --- teks vonis ----------------------------------------------------------------- #

def _label_scope(v: Verification) -> str:
    return "label" if v.full else "label pratinjau"


def _label_difference_text(v: Verification) -> str:
    text = f"{format_identical_pct(v.identical_pct)} % {_label_scope(v)} identik"
    if v.n_diff is not None and v.n_compared:
        text += f" ({format_pixels(v.n_diff)} dari {format_pixels(v.n_compared)} piksel berbeda)"
    return text


def verdict_text(v: Verification) -> str:
    """Kalimat vonis (tanpa ikon)."""
    if v.status == STATUS_NOT_COMPARABLE:
        return "Tidak sebanding: " + "; ".join(v.reasons)
    if v.status == STATUS_IDENTICAL:
        if v.full:
            return f"Label identik 100 % ({format_pixels(v.n_compared)} piksel), centroid sama"
        if v.n_compared:
            return (
                f"Verifikasi parsial: label pratinjau ({format_pixels(v.n_compared)} piksel pratinjau), "
                "counts, dan centroid sama. Label resolusi penuh belum dibandingkan"
            )
        return (
            "Verifikasi parsial: counts dan centroid sama. Label pratinjau tidak dibandingkan "
            "dan label resolusi penuh belum dibandingkan"
        )
    parts: list[str] = []
    if DIFF_LABELS in v.differences:
        parts.append(_label_difference_text(v))
    elif v.identical_pct is not None:
        parts.append(f"{_label_scope(v)} identik")
    if DIFF_CENTROIDS in v.differences:
        parts.append("centroid berbeda")
    if DIFF_COUNTS in v.differences:
        parts.append("counts berbeda")
    return "Berbeda: " + "; ".join(parts)


def verdict_style(v: Verification) -> tuple[str, str]:
    """(ikon, warna teks Streamlit) untuk vonis: hijau hanya untuk identik di resolusi penuh."""
    if v.status == STATUS_DIFFERENT:
        return ICON_BAD, "red"
    if v.status == STATUS_NOT_COMPARABLE:
        return ICON_WARN, "orange"
    return (ICON_OK, "green") if v.full else (ICON_PARTIAL, "blue")


def source_text(v: Verification) -> str | None:
    """Keterangan sumber; None bila tidak ada (tidak sebanding)."""
    if v.source == SOURCE_FILE:
        return f"Sumber: {VERIFICATION_FILE} (modul verifikasi)."
    if v.source == SOURCE_LABELS:
        return "Sumber: labels.u8 kedua run, resolusi penuh (dibandingkan per blok)."
    if v.source == SOURCE_PREVIEW:
        return (
            "Sumber: pratinjau (labels_preview.png, counts.json, centroids.json). Label resolusi "
            "penuh dibandingkan bila kedua run punya labels.u8 (landcover --write-labels) atau "
            f"salah satu punya {VERIFICATION_FILE}."
        )
    return None


def _yes_no(flag: bool | None, yes: str = "sama", no: str = "berbeda") -> str:
    return "tidak dibandingkan" if flag is None else (yes if flag else no)


def detail_rows(v: Verification) -> list[tuple[str, str]]:
    """Baris (pemeriksaan, hasil) untuk tabel rincian; kosong bila tidak sebanding."""
    if v.status == STATUS_NOT_COMPARABLE:
        return []
    if v.identical_pct is None:
        labels = "tidak dibandingkan"
    elif DIFF_LABELS in v.differences:
        labels = _label_difference_text(v)
    else:
        labels = f"identik ({format_pixels(v.n_compared)} piksel)"
    return [
        (_label_scope(v).capitalize(), labels),
        ("Centroid", _yes_no(v.centroids_equal)),
        ("Counts", _yes_no(v.counts_equal)),
        ("Sumber", v.source or "-"),
    ]


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

def _params_text(run: RunResult) -> str:
    m = run.meta
    return f"{m.get('dataset', '?')} · K={run.k} · {m.get('iters', '?')} iterasi · seed {m.get('seed', '?')}"


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
