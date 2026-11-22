"""Teks vonis verifikasi antarmode (logika murni, tanpa Streamlit; logika bandingnya di `landcover/verify.py`).

Diekstrak dari `ui/verify_badge.py`. Aturan kata (jangan dilonggarkan):
  * "Label identik 100 % (N piksel), centroid sama" HANYA untuk sumber yang menjangkau label resolusi
    penuh (`labels.u8` atau `verification.json`). Hasil dari pratinjau selalu berkata "Verifikasi
    parsial" dan hanya menyebut jumlah piksel pratinjau.
  * Persen identik dipotong ke bawah (bukan dibulatkan): satu piksel berbeda dari 120 juta tidak
    boleh tampil "100.00 %".
  * Hijau (`LEVEL_OK`) hanya untuk identik di resolusi penuh; pratinjau paling tinggi `LEVEL_PARTIAL`.
  * Format angka mengikuti aplikasi (ribuan koma, desimal titik, mis. 120,560,400 dan 99.98 %).
Streamlit dan API membaca kalimat yang sama dari sini; Streamlit memetakan `verdict_level` ke warna
teksnya sendiri (`ui.verify_badge.verdict_style`).
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

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
)

TITLE = "Status verifikasi antarmode"
MOCK_VERIFY_NOTICE = (
    "DATA TIRUAN: hasil verifikasi ini tidak membuktikan apa pun tentang core C++/OpenMP/CUDA."
)

ICON_OK, ICON_PARTIAL, ICON_BAD, ICON_WARN = "✅", "ℹ️", "❌", "⚠️"

LEVEL_OK, LEVEL_PARTIAL, LEVEL_BAD, LEVEL_WARN = "ok", "partial", "bad", "warn"
LEVEL_ICONS = {LEVEL_OK: ICON_OK, LEVEL_PARTIAL: ICON_PARTIAL, LEVEL_BAD: ICON_BAD, LEVEL_WARN: ICON_WARN}


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


def verdict_level(v: Verification) -> str:
    """Tingkat vonis: `LEVEL_OK` hanya untuk identik di resolusi penuh, `LEVEL_PARTIAL` untuk identik
    dari pratinjau, `LEVEL_BAD` untuk berbeda, `LEVEL_WARN` untuk tidak sebanding."""
    if v.status == STATUS_DIFFERENT:
        return LEVEL_BAD
    if v.status == STATUS_NOT_COMPARABLE:
        return LEVEL_WARN
    return LEVEL_OK if v.full else LEVEL_PARTIAL


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


def params_text(run: RunResult) -> str:
    m = run.meta
    return f"{m.get('dataset', '?')} · K={run.k} · {m.get('iters', '?')} iterasi · seed {m.get('seed', '?')}"


__all__ = [
    "ICON_BAD", "ICON_OK", "ICON_PARTIAL", "ICON_WARN", "LEVEL_BAD", "LEVEL_ICONS", "LEVEL_OK",
    "LEVEL_PARTIAL", "LEVEL_WARN", "MOCK_VERIFY_NOTICE", "TITLE", "detail_rows",
    "format_identical_pct", "format_pixels", "params_text", "source_text", "verdict_level",
    "verdict_text",
]
