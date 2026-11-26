"""Tes grafik waktu per tahap (`ui/charts.py`, plan Langkah 7), tanpa browser.

`stage_chart` mengembalikan `go.Figure`; tes membaca trace-nya langsung. Trace diberi
`meta={"kind": "stage"|"legend"|"total", ...}` supaya mudah dipilah. Folder run dibuat dengan
`make_run_dir` (conftest); tes pergeseran bottleneck memakai MockLauncher pada manifest tile penuh.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from landcover.contract import STAGES, StageTiming, read_run
from launchers.base import RunParams, out_dir_for
from launchers.mock import MockLauncher
from ui import charts
from ui import runs as run_state
from ui.charts import (
    DARK,
    DEVICE_COLORS,
    LIGHT,
    STAGE_NAMES,
    UNKNOWN_DEVICE_COLOR,
    format_duration,
    run_label,
    run_labels,
    stage_chart,
    stage_table,
)

CPU_LABEL = "CPU saja (L0)"
OPT_LABEL = "Optimalisasi · CUDA"
DEFAULT_SUM_S = 0.120 + 0.001 + 0.800 + 0.060 + 0.200       # jumlah tahap pada timing bawaan conftest


# --- pembantu ---------------------------------------------------------------- #

def opt_run(make_run_dir, **kw):
    """Run Optimalisasi CUDA (bawaan conftest): baca/pewarnaan/tulis omp, init serial, kmeans cuda."""
    return read_run(make_run_dir(**kw))


def cpu_run(make_run_dir, **kw):
    meta = {"mode": "cpu", "kmeans_device": "serial", "threads": 1, **kw.pop("meta", {})}
    timing = {"kmeans_ms": 40000.0, "read_ms": 8000.0, "write_ms": 11000.0, "color_ms": 700.0,
              "total_ms": 59701.0, "h2d_ms": 0.0, "kernel_ms": 0.0, "d2h_ms": 0.0, **kw.pop("timing", {})}
    return read_run(make_run_dir(meta=meta, timing=timing, **kw))


def pair(make_run_dir):
    cpu, opt = cpu_run(make_run_dir), opt_run(make_run_dir)
    return [cpu, opt], [CPU_LABEL, OPT_LABEL]


def of_kind(fig, kind):
    return [t for t in fig.data if t.meta and t.meta.get("kind") == kind]


def stage_traces(fig):
    return {t.meta["stage"]: t for t in of_kind(fig, "stage")}


def shares(fig, index):
    """{tahap: nilai x} untuk bar ke-`index`."""
    return {s: t.x[index] for s, t in stage_traces(fig).items()}


def with_stage(run, stage, timing):
    return replace(run, stages={**run.stages, stage: timing})


def luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# --- identitas visual -------------------------------------------------------- #

def test_nama_perangkat_satu_sumber_dengan_ui_runs():
    assert charts.DEVICE_NAMES is run_state.DEVICE_LABELS          # objek yang sama, bukan salinan
    assert set(DEVICE_COLORS) == set(charts.DEVICE_NAMES) == {"serial", "omp", "cuda"}


def test_warna_perangkat_sesuai_plan_dan_dipakai_laporan():
    assert DEVICE_COLORS == {"serial": "#8C8C8C", "omp": "#2F6FDE", "cuda": "#4E9A06"}
    assert UNKNOWN_DEVICE_COLOR not in DEVICE_COLORS.values()


def test_tulisan_putih_cukup_kontras_di_semua_warna_perangkat():
    assert charts.SEGMENT_TEXT == "#ffffff"      # APCA |Lc| 67-78 (hitam hanya 32-43) pada ketiga warna
    for color in DEVICE_COLORS.values():
        assert contrast(charts.SEGMENT_TEXT, color) >= 3.3, color


def test_nama_tahap_mencakup_semua_tahap_kontrak():
    assert set(STAGE_NAMES) == set(STAGES)


# --- struktur grafik --------------------------------------------------------- #

def test_satu_run_lima_tahap_urut_legenda_perangkat_dan_total(make_run_dir):
    run = opt_run(make_run_dir)
    fig = stage_chart([run], [OPT_LABEL])

    stages = of_kind(fig, "stage")
    assert [t.meta["stage"] for t in stages] == list(STAGES)         # urutan tumpukan = urutan tahap
    assert all(len(t.x) == 1 and list(t.y) == [OPT_LABEL] and t.orientation == "h" for t in stages)
    assert [t.x[0] for t in stages] == pytest.approx([0.120, 0.001, 0.800, 0.060, 0.200])
    assert not any(t.showlegend for t in stages)                     # legenda = perangkat, bukan tahap

    legend = of_kind(fig, "legend")
    assert [t.meta["device"] for t in legend] == ["serial", "omp", "cuda"]   # urutan tetap
    assert [t.name for t in legend] == ["CPU serial", "OpenMP", "CUDA"]
    assert all(t.showlegend and t.x == (None,) for t in legend)      # entri legenda saja, tanpa panjang
    assert fig.layout.barmode == "stack"

    (total,) = of_kind(fig, "total")
    assert tuple(total.x) == (run.total,) == (1.2,)
    assert total.marker.symbol == "line-ns"                          # bentuk beda dari segmen (bukan abu-abu serial)
    assert total.marker.color != DEVICE_COLORS["serial"]
    assert total.text == ("total 1.20 s",)
    assert total.showlegend and total.name == charts.TOTAL_NAME     # penanda dijelaskan di legenda


def test_dua_run_warna_mengikuti_perangkat_per_tahap(make_run_dir):
    runs, labels = pair(make_run_dir)
    fig = stage_chart(runs, labels)
    by_stage = stage_traces(fig)

    cpu_colors = {s: t.marker.color[0] for s, t in by_stage.items()}
    assert set(cpu_colors.values()) == {DEVICE_COLORS["serial"]}     # CPU saja: semua serial
    opt_colors = {s: t.marker.color[1] for s, t in by_stage.items()}
    assert opt_colors == {
        "baca": DEVICE_COLORS["omp"], "inisialisasi": DEVICE_COLORS["serial"],
        "kmeans": DEVICE_COLORS["cuda"], "pewarnaan": DEVICE_COLORS["omp"], "tulis": DEVICE_COLORS["omp"],
    }
    assert all(list(t.y) == labels for t in by_stage.values())
    assert shares(fig, 0)["kmeans"] == pytest.approx(40.0) and shares(fig, 1)["kmeans"] == pytest.approx(0.8)


def test_legenda_hanya_perangkat_yang_muncul(make_run_dir):
    fig = stage_chart([cpu_run(make_run_dir)], [CPU_LABEL])
    assert [t.meta["device"] for t in of_kind(fig, "legend")] == ["serial"]


def test_urutan_sumbu_y_run_pertama_di_atas(make_run_dir):
    runs, labels = pair(make_run_dir)
    fig = stage_chart(runs, labels)
    assert fig.layout.yaxis.autorange == "reversed"
    assert list(fig.layout.yaxis.categoryarray) == labels


def test_geometri_bar_tipis_dan_tinggi_menyertakan_sumbu(make_run_dir):
    runs, labels = pair(make_run_dir)
    one = stage_chart(runs[:1], labels[:1])
    two = stage_chart(runs, labels)
    assert charts.ROW_PX * (1 - charts.BARGAP) <= 24 + 1e-9         # tebal bar <= 24 px (skill dataviz)
    assert one.layout.height == charts.ROW_PX + charts.CHROME_PX
    assert two.layout.height == 2 * charts.ROW_PX + charts.CHROME_PX
    assert two.layout.height > one.layout.height
    assert two.layout.margin.t + two.layout.margin.b <= charts.CHROME_PX     # sumbu x tidak terpotong


def test_label_segmen_horizontal_putih_dan_disembunyikan_bila_tak_muat(make_run_dir):
    fig = stage_chart(*pair(make_run_dir))
    assert fig.layout.uniformtext.mode == "hide" and fig.layout.uniformtext.minsize == charts.MIN_TEXT_PX
    for t in of_kind(fig, "stage"):
        assert (t.textposition, t.textangle, t.textfont.color) == ("inside", 0, charts.SEGMENT_TEXT)
        assert t.marker.line.width == charts.GAP_PX                  # jarak 2 px antar segmen
        assert t.marker.line.color == LIGHT.surface


def test_teks_segmen_nama_tahap_dan_durasi_kosong_bila_nol(make_run_dir):
    run = with_stage(opt_run(make_run_dir), "inisialisasi", StageTiming(0.0, "serial", None))
    fig = stage_chart([run], [OPT_LABEL])
    by_stage = stage_traces(fig)
    assert by_stage["kmeans"].text == ("K-Means 800 ms",)
    assert by_stage["baca"].text == ("Baca 120 ms",)
    assert by_stage["inisialisasi"].text == ("",)                    # tidak ada tulisan untuk segmen selebar nol


# --- total dari timing.json -------------------------------------------------- #

@pytest.mark.parametrize("total_ms, label", [(5000.0, "lebih besar"), (500.0, "lebih kecil")])
def test_total_dari_timing_bukan_jumlah_segmen(make_run_dir, total_ms, label):
    run = read_run(make_run_dir(timing={"total_ms": total_ms}))
    assert (total_ms / 1000 > DEFAULT_SUM_S) == (label == "lebih besar")
    fig = stage_chart([run], [OPT_LABEL])
    (total,) = of_kind(fig, "total")
    assert tuple(total.x) == (total_ms / 1000,)                      # persis timing.total, tidak dijumlah ulang
    assert sum(t.x[0] for t in of_kind(fig, "stage")) == pytest.approx(DEFAULT_SUM_S)
    longest = max(total_ms / 1000, DEFAULT_SUM_S)
    assert fig.layout.xaxis.range[0] == 0
    assert fig.layout.xaxis.range[1] >= longest * charts.HEADROOM     # ruang untuk penanda dan tulisan total
    assert charts.HEADROOM >= 1.1


def test_total_tidak_valid_tidak_digambar_dan_tidak_error(make_run_dir):
    run = replace(opt_run(make_run_dir), total=float("nan"))
    fig = stage_chart([run], [OPT_LABEL])
    (total,) = of_kind(fig, "total")
    assert total.x == (None,) and total.text == ("",)
    assert fig.layout.xaxis.range[1] >= DEFAULT_SUM_S


# --- hover ------------------------------------------------------------------- #

def test_hover_kmeans_cuda_memuat_h2d_kernel_d2h_saja(make_run_dir):
    runs, labels = pair(make_run_dir)
    by_stage = stage_traces(stage_chart(runs, labels))
    cpu_hover, opt_hover = by_stage["kmeans"].hovertext
    for key in ("h2d", "kernel", "d2h"):
        assert key not in cpu_hover                                  # K-Means serial: tidak ada rincian transfer
        assert key in opt_hover
    assert "kernel: 700 ms" in opt_hover and "h2d: 20 ms" in opt_hover and "d2h: 10 ms" in opt_hover
    assert "CUDA" in opt_hover and "CPU serial" in cpu_hover
    for stage in set(STAGES) - {"kmeans"}:
        assert "kernel" not in "".join(by_stage[stage].hovertext)
    assert by_stage["kmeans"].hovertemplate == "%{hovertext}<extra></extra>"


def test_hover_kmeans_openmp_tanpa_rincian(make_run_dir):
    run = read_run(make_run_dir(meta={"kmeans_device": "omp"}))
    (hover,) = stage_traces(stage_chart([run], ["opt"]))["kmeans"].hovertext
    assert "OpenMP" in hover and "kernel" not in hover and "h2d" not in hover


# --- tampilan persen --------------------------------------------------------- #

def test_persen_menjumlah_100_per_bar_dan_total_resmi_sebagai_teks(make_run_dir):
    runs, labels = pair(make_run_dir)
    fig = stage_chart(runs, labels, as_percent=True)
    for i in range(2):
        assert sum(shares(fig, i).values()) == pytest.approx(100.0)
    (total,) = of_kind(fig, "total")
    assert tuple(total.x) == (100.0, 100.0) and total.mode == "text"      # bukan penanda pada sumbu persen
    assert total.text == ("total 59.70 s", "total 1.20 s")
    assert fig.layout.xaxis.range[1] > 100.0 and list(fig.layout.xaxis.tickvals) == [0, 25, 50, 75, 100]
    assert all("%" in t.text[0] for t in of_kind(fig, "stage") if t.x[0] >= 1)
    assert "persen" in fig.layout.title.text.lower() or "%" in fig.layout.title.text


def test_persen_tidak_membagi_dengan_nol(make_run_dir):
    run = opt_run(make_run_dir)
    for stage in STAGES:
        run = with_stage(run, stage, StageTiming(0.0, "serial", None))
    fig = stage_chart([run], [OPT_LABEL], as_percent=True)
    assert all(t.x[0] == 0.0 for t in of_kind(fig, "stage"))
    fig_s = stage_chart([run], [OPT_LABEL])
    assert fig_s.layout.xaxis.range[1] > 0                           # sumbu tidak runtuh bila semua nol


def test_pergeseran_bottleneck_pada_data_tiruan_tile_penuh(tmp_path):
    """K-Means dominan di CPU saja; baca/tulis dominan setelah optimalisasi (plan Langkah 7, "Cek")."""
    data, runs_dir = tmp_path / "data", tmp_path / "runs"
    (data / "uji").mkdir(parents=True)
    (data / "uji" / "manifest.json").write_text(
        json.dumps({"width": 10980, "height": 10980, "pixel_size_m": 10}), encoding="utf-8")
    launcher = MockLauncher(data, max_side=48, sleep=0)
    results = []
    for mode, kmeans, threads in (("cpu", "auto", 1), ("opt", "cuda", 8)):
        p = RunParams("uji", mode, kmeans, 8, 20, 42, threads)
        results.append(read_run(launcher.run(p, out_dir_for(runs_dir, "mock", p))))
    cpu, opt = results
    assert cpu.meta["width"] == 10980 and opt.kmeans_device == "cuda"

    fig = stage_chart(results, ["cpu", "opt"], as_percent=True)
    cpu_share, opt_share = shares(fig, 0), shares(fig, 1)
    assert max(cpu_share, key=cpu_share.get) == "kmeans" and cpu_share["kmeans"] > 50
    assert max(opt_share, key=opt_share.get) in {"baca", "tulis"}       # pergeseran bottleneck
    assert opt_share["kmeans"] < cpu_share["kmeans"] / 2
    assert opt_share["baca"] + opt_share["tulis"] > 50


# --- lencana tiruan, tema, ketahanan ------------------------------------------ #

@pytest.mark.parametrize("as_percent", [False, True])
def test_lencana_data_tiruan_di_judul_grafik(make_run_dir, as_percent):
    mock = read_run(make_run_dir(meta={"mock": True}))
    real = read_run(make_run_dir(meta={"mock": False}))
    assert "DATA TIRUAN" in stage_chart([mock], ["a"], as_percent).layout.title.text
    assert "DATA TIRUAN" not in stage_chart([real], ["a"], as_percent).layout.title.text
    assert "DATA TIRUAN" in stage_chart([real, mock], ["a", "b"], as_percent).layout.title.text   # salah satu tiruan = ditandai


def test_tema_gelap_mengganti_tinta_dan_permukaan_bukan_warna_perangkat(make_run_dir):
    runs, labels = pair(make_run_dir)
    light, dark = stage_chart(runs, labels), stage_chart(runs, labels, dark=True)
    assert light.layout.paper_bgcolor == LIGHT.surface and dark.layout.paper_bgcolor == DARK.surface
    assert dark.layout.xaxis.gridcolor == DARK.grid and light.layout.xaxis.gridcolor == LIGHT.grid
    assert of_kind(dark, "total")[0].marker.color == DARK.ink != of_kind(light, "total")[0].marker.color
    for a, b in zip(of_kind(light, "stage"), of_kind(dark, "stage")):
        assert list(a.marker.color) == list(b.marker.color)          # identitas perangkat sama di kedua tema
        assert b.marker.line.color == DARK.surface                   # jarak antar segmen mengikuti permukaan
    assert [t.marker.color for t in of_kind(light, "legend")] == [t.marker.color for t in of_kind(dark, "legend")]


def test_tema_aktif_dibaca_dari_streamlit_dengan_cadangan_terang(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(charts.st, "context", SimpleNamespace(theme=SimpleNamespace(type="dark")))
    assert charts.is_dark_theme() is True
    monkeypatch.setattr(charts.st, "context", SimpleNamespace(theme=SimpleNamespace(type="light")))
    assert charts.is_dark_theme() is False
    monkeypatch.setattr(charts.st, "context", SimpleNamespace())          # tanpa atribut theme: tidak boleh mogok
    assert charts.is_dark_theme() is False


def test_perangkat_tak_dikenal_tetap_tergambar(make_run_dir):
    run = with_stage(opt_run(make_run_dir), "kmeans", StageTiming(0.5, "tpu", None))
    fig = stage_chart([run], [OPT_LABEL])                            # tidak boleh KeyError
    kmeans = stage_traces(fig)["kmeans"]
    assert kmeans.marker.color == (UNKNOWN_DEVICE_COLOR,) and "tpu" in kmeans.hovertext[0]
    legend = of_kind(fig, "legend")
    assert [t.meta["device"] for t in legend] == ["serial", "omp", "tpu"]    # dikenal dulu, lalu yang aneh
    assert legend[-1].name == "tpu" and legend[-1].marker.color == UNKNOWN_DEVICE_COLOR


def test_waktu_nan_negatif_dan_bukan_angka_selebar_nol(make_run_dir):
    run = opt_run(make_run_dir)
    run = with_stage(run, "baca", StageTiming(float("nan"), "omp", None))
    run = with_stage(run, "tulis", StageTiming(-3.0, "omp", None))
    run = with_stage(run, "pewarnaan", StageTiming(float("inf"), "omp", None))
    fig = stage_chart([run], [OPT_LABEL])
    by_stage = stage_traces(fig)
    assert [by_stage[s].x[0] for s in ("baca", "tulis", "pewarnaan")] == [0.0, 0.0, 0.0]
    assert all(math.isfinite(t.x[0]) and t.x[0] >= 0 for t in of_kind(fig, "stage"))
    assert math.isfinite(fig.layout.xaxis.range[1])
    table = stage_table([run], [OPT_LABEL])
    assert table["Waktu (s)"].iloc[:5].min() >= 0


def test_masukan_salah_ditolak_dengan_jelas(make_run_dir):
    runs, labels = pair(make_run_dir)
    with pytest.raises(ValueError, match="minimal satu"):
        stage_chart([], [])
    with pytest.raises(ValueError, match="jumlah label"):
        stage_chart(runs, labels[:1])
    with pytest.raises(ValueError, match="unik"):
        stage_chart(runs, ["sama", "sama"])                           # label kembar = dua run tertumpuk di satu bar
    with pytest.raises(ValueError, match="unik"):
        stage_table(runs, ["sama", "sama"])


# --- pembantu tampilan --------------------------------------------------------- #

@pytest.mark.parametrize("seconds, text", [
    (None, "0 ms"), (0.0, "0 ms"), (-1.0, "0 ms"), (float("nan"), "0 ms"), (float("inf"), "0 ms"),
    (0.0004, "0.40 ms"), (0.0008, "0.80 ms"), (0.005, "5.00 ms"), (0.0099, "9.90 ms"), (0.01, "10 ms"), (0.45, "450 ms"), (0.9994, "999 ms"),
    (0.9996, "1.00 s"), (1.0, "1.00 s"), (9.254, "9.25 s"), (59.5249, "59.52 s"),
])
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


def test_label_dan_urutan_run(make_run_dir):
    cpu, opt = cpu_run(make_run_dir), opt_run(make_run_dir)
    assert run_label(cpu) == CPU_LABEL == run_state.MODE_LABELS["cpu"]
    assert run_label(opt) == OPT_LABEL
    assert run_label(read_run(make_run_dir(meta={"kmeans_device": "omp"}))) == "Optimalisasi · OpenMP"
    assert run_labels(cpu, opt) == ([cpu, opt], [CPU_LABEL, OPT_LABEL])
    assert run_labels(None, opt) == ([opt], [OPT_LABEL])
    assert run_labels(cpu, None) == ([cpu], [CPU_LABEL])
    assert run_labels(None, None) == ([], [])


def test_tabel_kembaran_grafik(make_run_dir):
    runs, labels = pair(make_run_dir)
    df = stage_table(runs, labels)
    assert list(df.columns) == ["Run", "Tahap", "Perangkat", "Waktu (s)", "Persen dari jumlah tahap"]
    assert len(df) == 2 * (len(STAGES) + 1)
    for run, label in zip(runs, labels):
        part = df[df["Run"] == label]
        assert list(part["Tahap"]) == [STAGE_NAMES[s] for s in STAGES] + [charts.TOTAL_NAME]
        stage_rows, total_row = part.iloc[:-1], part.iloc[-1]
        assert stage_rows["Persen dari jumlah tahap"].sum() == pytest.approx(100.0)
        assert total_row["Waktu (s)"] == run.total                     # total resmi, bukan jumlah tahap
        assert stage_rows["Waktu (s)"].tolist() == pytest.approx([run.stages[s].time for s in STAGES])
    assert set(df[df["Run"] == CPU_LABEL]["Perangkat"].iloc[:-1]) == {"CPU serial"}
    assert df[df["Run"] == OPT_LABEL].iloc[2]["Perangkat"] == "CUDA"        # baris K-Means
