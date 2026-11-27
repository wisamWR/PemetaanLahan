"""Data grafik waktu per tahap (logika murni, tanpa Streamlit, plotly, atau pandas).

Diekstrak dari `ui/charts.py`: identitas perangkat (warna, nama), nama tahap, pemformatan durasi,
sel tahap per run (detik, perangkat, pangsa), dan label run. Dipakai bersama oleh Streamlit
(`ui/charts.py` menggambar dengan plotly) dan API (frontend React menggambar dengan Recharts dari
angka yang sama). Aturan angka tetap di sini: waktu NaN/negatif dihitung nol, pangsa dihitung dari
JUMLAH waktu tahap (bukan dari total resmi `timing.json`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from landcover.contract import STAGES, RunResult
from services.runs import DEVICE_LABELS, MODE_LABELS

# Warna dari plan Langkah 7. Divalidasi dengan skill dataviz (validate_palette.js, 21 Nov):
# lulus pita kecerahan, pemisahan CVD (terburuk dE 11.9 pada --pairs all), lantai penglihatan
# normal (dE 18.1), dan kontras >= 3:1 pada permukaan terang dan gelap. SATU-SATUNYA yang tidak
# lulus: lantai kroma abu-abu serial. Itu disengaja: serial = garis dasar tanpa percepatan, jadi
# harus tampak netral. Identitas tetap terbawa legenda, tulisan di dalam segmen, hover, dan tabel.
DEVICE_COLORS = {"serial": "#8C8C8C", "omp": "#2F6FDE", "cuda": "#4E9A06"}
DEVICE_NAMES = DEVICE_LABELS            # objek yang SAMA dengan ui.runs: satu sumber nama perangkat
UNKNOWN_DEVICE_COLOR = "#B5488A"        # perangkat tak dikenal (hasil janggal tetap tergambar)
DEVICE_ORDER = tuple(DEVICE_COLORS)     # urutan legenda: serial, omp, cuda

STAGE_NAMES = {
    "baca": "Baca",
    "inisialisasi": "Inisialisasi",
    "kmeans": "K-Means",
    "pewarnaan": "Pewarnaan",
    "tulis": "Tulis",
}

TOTAL_NAME = "Total resmi (timing.json)"


def format_duration(seconds: float | None) -> str:
    """Durasi untuk label: milidetik di bawah 1 s ("0.80 ms", "450 ms"), selain itu "12.34 s"."""
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "0 ms"
    if seconds < 0.01:
        return f"{seconds * 1000:.2f} ms"
    if round(seconds * 1000) < 1000:         # 0.9996 s dibulatkan ke "1.00 s", bukan "1000 ms"
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def clean_seconds(value: object) -> float:
    """Detik yang aman digambar: NaN, tak hingga, negatif, atau bukan angka -> 0 (lebar nol)."""
    try:
        v = float(value)               # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def clean_seconds_or_none(value: object) -> float | None:
    try:
        v = float(value)               # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def device_color(device: str) -> str:
    return DEVICE_COLORS.get(device, UNKNOWN_DEVICE_COLOR)


def device_name(device: str) -> str:
    """Nama tampilan; perangkat tak dikenal ditampilkan apa adanya."""
    return DEVICE_NAMES.get(device, str(device))



@dataclass(frozen=True)
class StageCell:
    stage: str
    seconds: float
    device: str
    detail: dict[str, float] | None
    share: float                # persen dari jumlah waktu tahap run ini (0 bila jumlahnya 0)


def run_cells(run: RunResult) -> dict[str, StageCell]:
    raw = {s: clean_seconds(run.stages[s].time) for s in STAGES}
    total = sum(raw.values())
    cells: dict[str, StageCell] = {}
    for s in STAGES:
        st_ = run.stages[s]
        share = raw[s] / total * 100.0 if total > 0 else 0.0
        cells[s] = StageCell(s, raw[s], str(st_.device), st_.detail, share)
    return cells


def check_chart_inputs(runs: Sequence[RunResult], labels: Sequence[str]) -> None:
    if not runs:
        raise ValueError("stage_chart butuh minimal satu run")
    if len(runs) != len(labels):
        raise ValueError(f"jumlah label ({len(labels)}) harus sama dengan jumlah run ({len(runs)})")
    if len(set(labels)) != len(labels):
        # label kembar membuat dua run tertumpuk di satu bar tanpa tanda
        raise ValueError(f"label run harus unik: {list(labels)}")


def run_label(run: RunResult) -> str:
    """Label pendek satu bar: mode (+ perangkat K-Means untuk Optimalisasi)."""
    if run.mode == "cpu":
        return MODE_LABELS["cpu"]
    mode = MODE_LABELS.get(run.mode, str(run.mode))
    return f"{mode} · {device_name(run.kmeans_device)}"


def run_labels(cpu: RunResult | None, opt: RunResult | None) -> tuple[list[RunResult], list[str]]:
    """Run yang digambar (CPU saja di atas, lalu Optimalisasi) beserta labelnya; yang None dilewati."""
    runs = [r for r in (cpu, opt) if r is not None]
    return runs, [run_label(r) for r in runs]


def share_text(share: float) -> str:
    """Pangsa tahap untuk label: "36%" bila >= 10, selain itu satu desimal ("3.5%")."""
    return f"{share:.0f}%" if share >= 10 else f"{share:.1f}%"


def stage_records(runs: Sequence[RunResult], labels: Sequence[str]) -> list[dict]:
    """Baris tabel grafik: satu per run x tahap, plus baris total resmi per run.

    Semua nilai grafik terbaca di sini, juga label yang tidak muat di segmen. Dipakai
    `ui.charts.stage_table` (DataFrame) dan API."""
    runs, labels = list(runs), list(labels)
    check_chart_inputs(runs, labels)
    records: list[dict] = []
    for run, label in zip(runs, labels):
        for stage, cell in run_cells(run).items():
            records.append({
                "Run": label, "Tahap": STAGE_NAMES[stage], "Perangkat": device_name(cell.device),
                "Waktu (s)": cell.seconds, "Persen dari jumlah tahap": cell.share,
            })
        total = clean_seconds_or_none(run.total)
        records.append({
            "Run": label, "Tahap": TOTAL_NAME, "Perangkat": "",
            "Waktu (s)": float("nan") if total is None else total,
            "Persen dari jumlah tahap": float("nan"),
        })
    return records


__all__ = [
    "DEVICE_COLORS", "DEVICE_NAMES", "DEVICE_ORDER", "STAGE_NAMES", "StageCell", "TOTAL_NAME",
    "UNKNOWN_DEVICE_COLOR", "check_chart_inputs", "clean_seconds", "clean_seconds_or_none",
    "device_color", "device_name", "format_duration", "run_cells", "run_label", "run_labels",
    "share_text", "stage_records",
]
