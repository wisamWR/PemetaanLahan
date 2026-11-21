"""Grafik waktu per tahap (plan Langkah 7).

Satu bar horizontal bertumpuk per run; segmen = tahap (baca, inisialisasi, K-Means, pewarnaan,
tulis), warna = perangkat yang menjalankan tahap itu (CPU serial / OpenMP / CUDA), legenda =
perangkat. Tampilan kedua (`as_percent=True`) menormalkan tiap bar ke 100 % supaya pergeseran
bottleneck terlihat (K-Means dominan di CPU saja, baca/tulis dominan setelah optimalisasi).

`stage_chart`, `stage_table`, `run_labels`, dan `format_duration` murni (teruji tanpa browser).
`render_stage_chart` / `render_stage_slot` hanya menggambar. Aturan visual mengikuti skill
`dataviz`: warna dipilih menurut tugasnya (identitas perangkat), bar tipis, jarak 2 px antar
segmen, label di dalam segmen hanya bila muat (sisanya di hover dan tabel), grid hairline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from landcover.contract import STAGES, RunResult
from ui.badges import MOCK_BADGE
from ui.compare import MOCK_NOTICE
from ui.runs import DEVICE_LABELS, MODE_LABELS

# --- identitas visual (satu sumber; laporan memakai nilai yang sama) -------- #

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

SEGMENT_TEXT = "#ffffff"        # tulisan di dalam segmen: putih di ketiga warna (APCA |Lc| 67-78)


@dataclass(frozen=True)
class ChartTheme:
    surface: str
    ink: str            # teks utama
    ink2: str           # teks sekunder (sumbu, legenda)
    grid: str           # garis grid hairline
    axis: str           # garis sumbu


LIGHT = ChartTheme(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", axis="#c3c2b7")
DARK = ChartTheme(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a", axis="#383835")

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# geometri: tinggi bar = ROW_PX * (1 - BARGAP) = 24 px (batas bar pada skill dataviz)
ROW_PX = 64
BARGAP = 0.625
GAP_PX = 2                  # jarak permukaan antar segmen
CHROME_PX = 156             # judul + legenda (atas) dan sumbu x (bawah)
MIN_TEXT_PX = 11            # tulisan yang butuh lebih kecil dari ini disembunyikan
HEADROOM = 1.15             # ruang di kanan bar untuk penanda dan tulisan total
PERCENT_TICKS = (0, 25, 50, 75, 100)

VIEW_SECONDS = "seconds"
VIEW_PERCENT = "percent"
VIEW_LABELS = {VIEW_SECONDS: "Detik", VIEW_PERCENT: "Persentase per tahap"}
KEY_VIEW = "chart_view"
CHART_CONFIG = {"displaylogo": False, "displayModeBar": "hover"}

TOTAL_NAME = "Total resmi (timing.json)"


# --- pembantu murni --------------------------------------------------------- #

def format_duration(seconds: float | None) -> str:
    """Durasi untuk label: milidetik di bawah 1 s ("0.80 ms", "450 ms"), selain itu "12.34 s"."""
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "0 ms"
    if seconds < 0.01:
        return f"{seconds * 1000:.2f} ms"
    if round(seconds * 1000) < 1000:         # 0.9996 s dibulatkan ke "1.00 s", bukan "1000 ms"
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def _clean(value: object) -> float:
    """Detik yang aman digambar: NaN, tak hingga, negatif, atau bukan angka -> 0 (lebar nol)."""
    try:
        v = float(value)               # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def _clean_or_none(value: object) -> float | None:
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


def _share_text(share: float) -> str:
    return f"{share:.0f}%" if share >= 10 else f"{share:.1f}%"


@dataclass(frozen=True)
class _Cell:
    stage: str
    seconds: float
    device: str
    detail: dict[str, float] | None
    share: float                # persen dari jumlah waktu tahap run ini (0 bila jumlahnya 0)


def _run_cells(run: RunResult) -> dict[str, _Cell]:
    raw = {s: _clean(run.stages[s].time) for s in STAGES}
    total = sum(raw.values())
    cells: dict[str, _Cell] = {}
    for s in STAGES:
        st_ = run.stages[s]
        share = raw[s] / total * 100.0 if total > 0 else 0.0
        cells[s] = _Cell(s, raw[s], str(st_.device), st_.detail, share)
    return cells


def _hover(label: str, cell: _Cell) -> str:
    lines = [
        f"<b>{label}</b>",
        f"{STAGE_NAMES[cell.stage]}: {format_duration(cell.seconds)}",
        f"Perangkat: {device_name(cell.device)}",
        f"{cell.share:.1f}% dari jumlah waktu tahap",
    ]
    if cell.detail:
        for key in ("h2d", "kernel", "d2h"):
            if key in cell.detail:
                lines.append(f"{key}: {format_duration(_clean(cell.detail[key]))}")
    return "<br>".join(lines)


def _check_inputs(runs: Sequence[RunResult], labels: Sequence[str]) -> None:
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


def is_dark_theme() -> bool:
    """Tema aktif Streamlit gelap? Cadangan terang bila tidak bisa dibaca (mis. di AppTest)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:   # noqa: BLE001 (st.context.theme tidak tersedia di semua konteks)
        return False


# --- grafik ----------------------------------------------------------------- #

def stage_chart(
    runs: Sequence[RunResult],
    labels: Sequence[str],
    as_percent: bool = False,
    *,
    dark: bool = False,
) -> go.Figure:
    """Bar horizontal bertumpuk, satu bar per run.

    Segmen = tahap (urutan `STAGES`), warna = perangkat tahap itu. Penanda total memakai
    `run.total` dari `timing.json` (bukan jumlah segmen: pada L3 tahap bisa tumpang tindih dan
    di WSL jumlah bisa melebihi total). `as_percent=True` menormalkan tiap bar ke 100 % dari
    JUMLAH waktu tahap (jadi bukan dari total resmi, yang boleh berbeda); total resmi tetap
    ditulis sebagai teks di ujung bar. Waktu NaN/negatif digambar selebar nol.

    ValueError bila `runs` kosong, jumlah `labels` berbeda, atau ada label kembar."""
    runs, labels = list(runs), list(labels)
    _check_inputs(runs, labels)
    theme = DARK if dark else LIGHT
    rows = [_run_cells(r) for r in runs]
    totals = [_clean_or_none(r.total) for r in runs]
    sums = [sum(c.seconds for c in row.values()) for row in rows]
    mock = any(r.mock for r in runs)

    fig = go.Figure()

    # 1. satu trace per tahap (urutan tumpukan = urutan tahap), warna per titik = perangkat
    for stage in STAGES:
        cells = [row[stage] for row in rows]
        fig.add_trace(go.Bar(
            x=[c.share if as_percent else c.seconds for c in cells],
            y=labels,
            orientation="h",
            name=STAGE_NAMES[stage],
            marker=dict(
                color=[device_color(c.device) for c in cells],
                line=dict(color=theme.surface, width=GAP_PX),
            ),
            text=[
                "" if c.seconds <= 0 else
                f"{STAGE_NAMES[stage]} {_share_text(c.share) if as_percent else format_duration(c.seconds)}"
                for c in cells
            ],
            textposition="inside",
            insidetextanchor="middle",
            textangle=0,
            textfont=dict(color=SEGMENT_TEXT, size=12),
            hovertext=[_hover(lab, c) for lab, c in zip(labels, cells)],
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
            meta={"kind": "stage", "stage": stage},
        ))

    # 2. legenda = perangkat yang muncul (trace kosong hanya untuk entri legenda)
    present = {c.device for row in rows for c in row.values()}
    ordered = [d for d in DEVICE_ORDER if d in present] + sorted(present - set(DEVICE_ORDER))
    for device in ordered:
        fig.add_trace(go.Bar(
            x=[None], y=[labels[0]], orientation="h",
            name=device_name(device),
            marker=dict(color=device_color(device)),
            legendgroup=f"device-{device}",
            showlegend=True,
            hoverinfo="skip",
            meta={"kind": "legend", "device": device},
        ))

    # 3. total resmi
    total_text = [f"total {format_duration(t)}" if t is not None else "" for t in totals]
    if as_percent:
        # sumbu persen: total resmi hanya sebagai teks di ujung bar (bar selalu berakhir di 100)
        fig.add_trace(go.Scatter(
            x=[100.0] * len(runs), y=labels, mode="text",
            text=total_text, textposition="middle right",
            textfont=dict(color=theme.ink, size=13), cliponaxis=False,
            showlegend=False, hoverinfo="skip",
            meta={"kind": "total"},
        ))
        upper = 100.0 * HEADROOM
    else:
        fig.add_trace(go.Scatter(
            x=totals, y=labels, mode="markers+text",
            name=TOTAL_NAME,
            marker=dict(symbol="line-ns", size=28, color=theme.ink, line=dict(color=theme.ink, width=3)),
            text=total_text, textposition="middle right",
            textfont=dict(color=theme.ink, size=13), cliponaxis=False,
            hovertext=[f"<b>{lab}</b><br>Total resmi: {format_duration(t)}" if t is not None else ""
                       for lab, t in zip(labels, totals)],
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=True,
            meta={"kind": "total"},
        ))
        longest = max([*sums, *[t for t in totals if t is not None]], default=0.0)
        upper = longest * HEADROOM if longest > 0 else 1.0

    title = "Pangsa waktu per tahap (%)" if as_percent else "Waktu per tahap (detik)"
    if mock:
        title += "  ·  <b>DATA TIRUAN</b>"
    xaxis = dict(
        range=[0, upper], fixedrange=True,
        title=dict(text="% dari jumlah waktu tahap" if as_percent else "Waktu (detik)",
                   font=dict(color=theme.ink2)),
        showgrid=True, gridcolor=theme.grid, gridwidth=1, zeroline=False,
        showline=True, linecolor=theme.axis, linewidth=1,
        tickfont=dict(color=theme.ink2),
    )
    if as_percent:
        xaxis.update(tickmode="array", tickvals=list(PERCENT_TICKS), ticksuffix="%")
    else:
        xaxis.update(ticksuffix=" s")
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=1, yanchor="top", yref="container",
                   font=dict(size=16, color=theme.ink), pad=dict(t=6)),
        barmode="stack", bargap=BARGAP,
        uniformtext=dict(minsize=MIN_TEXT_PX, mode="hide"),
        paper_bgcolor=theme.surface, plot_bgcolor=theme.surface,
        font=dict(family=FONT_FAMILY, size=13, color=theme.ink2),
        height=len(runs) * ROW_PX + CHROME_PX,
        margin=dict(l=8, r=120, t=92, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(color=theme.ink2), bgcolor="rgba(0,0,0,0)"),
        xaxis=xaxis,
        yaxis=dict(
            autorange="reversed", categoryorder="array", categoryarray=labels,
            automargin=True, fixedrange=True, showgrid=False,
            showline=True, linecolor=theme.axis, linewidth=1, ticks="",
            tickfont=dict(color=theme.ink2),
        ),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
    )
    return fig


def stage_table(runs: Sequence[RunResult], labels: Sequence[str]) -> pd.DataFrame:
    """Kembaran tabel grafik (satu baris per run x tahap, plus baris total resmi per run).

    Semua nilai grafik terbaca di sini, juga label yang tidak muat di segmen."""
    runs, labels = list(runs), list(labels)
    _check_inputs(runs, labels)
    records: list[dict] = []
    for run, label in zip(runs, labels):
        for stage, cell in _run_cells(run).items():
            records.append({
                "Run": label, "Tahap": STAGE_NAMES[stage], "Perangkat": device_name(cell.device),
                "Waktu (s)": cell.seconds, "Persen dari jumlah tahap": cell.share,
            })
        total = _clean_or_none(run.total)
        records.append({
            "Run": label, "Tahap": TOTAL_NAME, "Perangkat": "",
            "Waktu (s)": float("nan") if total is None else total,
            "Persen dari jumlah tahap": float("nan"),
        })
    return pd.DataFrame.from_records(records)


# --- tampilan Streamlit ------------------------------------------------------ #

def _run_caption(run: RunResult) -> str:
    m = run.meta
    return f"{m.get('dataset', '?')} · K={run.k} · {m.get('iters', '?')} iterasi · seed {m.get('seed', '?')}"


def render_stage_chart(cpu: RunResult | None, opt: RunResult | None) -> None:
    """Grafik untuk pasangan (CPU saja, Optimalisasi); salah satu boleh None, jangan keduanya."""
    runs, labels = run_labels(cpu, opt)
    if not runs:
        raise ValueError("render_stage_chart butuh minimal satu run")
    mock = any(r.mock for r in runs)
    st.subheader(f"Waktu per tahap {MOCK_BADGE}" if mock else "Waktu per tahap")
    view = st.radio(
        "Tampilan grafik", list(VIEW_LABELS), key=KEY_VIEW, format_func=VIEW_LABELS.get, horizontal=True,
        help="Persentase menormalkan tiap bar ke 100 % supaya pergeseran bottleneck terlihat.",
    )
    as_percent = view == VIEW_PERCENT
    fig = stage_chart(runs, labels, as_percent=as_percent, dark=is_dark_theme())
    st.plotly_chart(fig, width="stretch", theme=None, key=f"chart_stage_{view}", config=CHART_CONFIG)

    st.caption(_run_caption(runs[0]))
    if as_percent:
        st.caption(
            "Persentase dihitung dari jumlah waktu tahap, bukan dari total resmi (total bisa berbeda "
            "bila tahap tumpang tindih). Tulisan yang tidak muat di segmen tidak ditampilkan: lihat hover atau tabel."
        )
    else:
        st.caption(
            "Warna = perangkat. Garis tegak = total resmi dari timing.json, bukan jumlah segmen. "
            "Tulisan yang tidak muat di segmen tidak ditampilkan: lihat hover atau tabel."
        )
    if mock:
        st.caption(MOCK_NOTICE)
    with st.expander("Tabel waktu per tahap"):
        st.dataframe(
            stage_table(runs, labels), hide_index=True, width="stretch",
            column_config={
                "Waktu (s)": st.column_config.NumberColumn(format="%.3f"),
                "Persen dari jumlah tahap": st.column_config.NumberColumn(format="%.1f"),
            },
        )


def render_stage_slot(cpu: RunResult | None, opt: RunResult | None) -> None:
    """Kotak grafik di halaman: grafik bila ada run, selain itu kotak kosong dengan petunjuk."""
    with st.container(border=True):
        if cpu is None and opt is None:
            st.caption("Grafik waktu per tahap: belum ada hasil (jalankan dulu)")
        else:
            render_stage_chart(cpu, opt)


__all__ = [
    "CHART_CONFIG", "DARK", "DEVICE_COLORS", "DEVICE_NAMES", "DEVICE_ORDER", "KEY_VIEW", "LIGHT",
    "STAGE_NAMES", "TOTAL_NAME", "UNKNOWN_DEVICE_COLOR", "VIEW_LABELS", "VIEW_PERCENT", "VIEW_SECONDS",
    "device_color", "device_name", "format_duration", "is_dark_theme", "render_stage_chart",
    "render_stage_slot", "run_label", "run_labels", "stage_chart", "stage_table",
]
