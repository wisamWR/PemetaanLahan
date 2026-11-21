"""Logika murni layar: sidebar (parameter), kartu perbandingan, dan peta (Langkah 6)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import write_run_dir
from landcover.contract import read_run
from landcover.labeling import UNNAMED
from landcover.render import class_palette, recolor
from launchers.base import RunParams, run_id
from ui import compare as C
from ui import maps as M
from ui.sidebar import FormValues, build_pair, build_params


def make_run(tmp_path, name, *, mode="opt", dataset="sleman", k=4, iters=20, seed=42, total_ms=1000.0, **meta):
    cpu = mode == "cpu"
    meta = {"dataset": dataset, "mode": mode, "k": k, "iters": iters, "seed": seed,
            "kmeans_device": "serial" if cpu else "cuda", "threads": 1 if cpu else 8, **meta}
    return read_run(write_run_dir(tmp_path / name, k=k, meta=meta, timing={"total_ms": total_ms}))


# --- sidebar: isian form -> RunParams ---------------------------------------- #

def test_build_params_memetakan_isian_form():
    v = FormValues("sleman", k=6, iters=15, mode="opt", kmeans="omp", threads=4, seed=7)
    p = build_params(v)
    assert p == RunParams("sleman", "opt", "omp", 6, 15, 7, 4)


def test_nilai_bawaan_sesuai_plan():
    v = FormValues("x")
    assert (v.k, v.iters, v.seed, v.threads, v.mode, v.kmeans, v.force) == (8, 20, 42, 8, "cpu", "auto", False)


def test_build_pair_hanya_beda_mode():
    v = FormValues("sleman", k=5, iters=11, mode="opt", kmeans="cuda", threads=3, seed=9)
    cpu, opt = build_pair(v)
    assert (cpu.mode, opt.mode) == ("cpu", "opt")
    for field in ("dataset", "k", "iters", "seed", "threads", "kmeans"):
        assert getattr(cpu, field) == getattr(opt, field) == getattr(v, field)
    # run_id CPU tidak bergantung pada kmeans/thread, jadi hasil CPU bisa dipakai ulang
    assert run_id(cpu) == run_id(build_params(FormValues("sleman", k=5, iters=11, seed=9)))


@pytest.mark.parametrize("kw", [{"dataset": "a/b"}, {"dataset": ""}, {"k": 17}, {"k": 0}, {"iters": 0},
                                {"threads": 0}, {"seed": -1}])
def test_build_params_nilai_tidak_valid_melempar_value_error(kw):
    with pytest.raises(ValueError):
        build_params(FormValues(**{"dataset": "ok", **kw}))


# --- kartu perbandingan ------------------------------------------------------- #

def test_speedup():
    assert C.speedup(60.0, 10.0) == 6.0
    assert C.speedup(None, 10.0) is None and C.speedup(60.0, None) is None
    assert C.speedup(60.0, 0.0) is None and C.speedup(60.0, -1.0) is None
    assert C.speedup(0.0, 5.0) == 0.0


def test_format_seconds_dan_timer_text():
    assert C.format_seconds(12.3456) == "12.35 s"
    assert C.format_seconds(0) == "0.00 s"
    assert C.format_seconds(None) == C.EMPTY == "—"
    assert C.timer_text("CPU saja (L0)", 1.234) == "⏱ **CPU saja (L0)** berjalan: 1.2 s"


def test_caption_perangkat_menyebut_thread_hanya_untuk_openmp(tmp_path):
    cpu = make_run(tmp_path, "c", mode="cpu")
    omp = make_run(tmp_path, "o", kmeans_device="omp", threads=6)
    cuda = make_run(tmp_path, "g", kmeans_device="cuda", threads=0)
    assert C._device_caption(cpu) == "K-Means: CPU serial"
    assert C._device_caption(omp) == "K-Means: OpenMP, 6 thread"
    assert C._device_caption(cuda) == "K-Means: CUDA"


def test_render_timer_menulis_ke_placeholder():
    class Box:
        text = None

        def markdown(self, text):
            self.text = text

    box = Box()
    C.render_timer(box, "Optimalisasi", 2.0)
    assert box.text == C.timer_text("Optimalisasi", 2.0)


def test_comparable_runs_memasangkan_dataset_k_iterasi_seed_sama(tmp_path):
    cpu = make_run(tmp_path, "cpu", mode="cpu")
    opt = make_run(tmp_path, "opt")
    k5 = make_run(tmp_path, "k5", k=5)
    lain = make_run(tmp_path, "lain", dataset="kulonprogo")
    runs = {"cpu": cpu, "opt": opt, "k5": k5, "lain": lain}
    assert C.comparable_runs(runs, "opt") == (cpu, opt)
    assert C.comparable_runs(runs, "cpu") == (cpu, opt)
    assert C.comparable_runs(runs, "k5") == (None, k5)            # K lain: belum ada pasangan CPU
    assert C.comparable_runs(runs, "lain") == (None, lain)
    assert C.comparable_runs(runs, "tidak_ada") == (None, None)


@pytest.mark.parametrize("beda", [{"iters": 10}, {"seed": 1}])
def test_comparable_runs_tidak_memasangkan_iterasi_atau_seed_beda(tmp_path, beda):
    cpu = make_run(tmp_path, "cpu", mode="cpu")
    opt = make_run(tmp_path, "opt", **beda)
    assert C.comparable_runs({"cpu": cpu, "opt": opt}, "opt") == (None, opt)


def test_comparable_runs_memilih_run_terbaru_dan_aktif_diutamakan(tmp_path):
    cpu = make_run(tmp_path, "cpu", mode="cpu")
    opt_lama = make_run(tmp_path, "opt_lama", total_ms=900.0)
    opt_baru = make_run(tmp_path, "opt_baru", total_ms=800.0)
    runs = {"opt_lama": opt_lama, "cpu": cpu, "opt_baru": opt_baru}      # urutan = kebaruan
    assert C.comparable_runs(runs, "cpu") == (cpu, opt_baru)             # pasangan = yang terbaru
    assert C.comparable_runs(runs, "opt_lama") == (cpu, opt_lama)        # yang dipilih tetap dipakai


# --- peta --------------------------------------------------------------------- #

def test_class_map_image_centroid_adalah_map_preview(tmp_path):
    run = make_run(tmp_path, "r")
    img = M.class_map_image(run, {i: UNNAMED for i in range(run.k)}, M.VIEW_CENTROID)
    assert img is run.map_preview


def test_class_map_image_kelas_memakai_palet_kelas(tmp_path):
    run = make_run(tmp_path, "r", k=4)
    cmap = {0: "Air", 1: "Air", 2: "Vegetasi", 3: UNNAMED}
    img = M.class_map_image(run, cmap, M.VIEW_CLASS)
    assert img.shape == run.map_preview.shape and img.dtype == np.uint8
    np.testing.assert_array_equal(img, recolor(run.labels_preview, class_palette(cmap, run.centroids)))
    air = {tuple(img[run.labels_preview == c][0]) for c in (0, 1)}
    assert len(air) == 1                                   # cluster 0 dan 1 digabung: satu warna
    assert not np.array_equal(img, run.map_preview)        # sudah berbeda dari warna centroid


def test_class_map_image_semua_tanpa_nama_sama_dengan_warna_centroid(tmp_path):
    run = make_run(tmp_path, "r", k=4)
    img = M.class_map_image(run, {i: UNNAMED for i in range(4)}, M.VIEW_CLASS)
    np.testing.assert_array_equal(img, run.map_preview)


def test_class_map_image_view_tidak_dikenal():
    with pytest.raises(ValueError, match="view"):
        M.class_map_image(None, {}, "lain")


def test_find_original_preview_jpg_lalu_png_lepas(tmp_path):
    (tmp_path / "sleman").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "sleman" / "preview.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "kecil.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "sleman.png")        # preview.jpg lebih diutamakan
    assert M.find_original(tmp_path, "sleman") == tmp_path / "sleman" / "preview.jpg"
    assert M.find_original(tmp_path, "kecil") == tmp_path / "kecil.png"
    assert M.find_original(tmp_path, "tidak_ada") is None


def test_load_original_mengecilkan_dan_menolak_yang_terlalu_besar(tmp_path, monkeypatch):
    Image.new("RGB", (400, 200), (9, 8, 7)).save(tmp_path / "a.png")
    out = M.load_original(tmp_path / "a.png", max_side=100)
    assert out.shape == (50, 100, 3) and out.dtype == np.uint8
    assert M.load_original(tmp_path / "a.png", max_side=1000).shape == (200, 400, 3)   # tidak diperbesar
    monkeypatch.setattr(M, "ORIGINAL_MAX_PIXELS", 1000)
    with pytest.raises(ValueError, match="terlalu besar"):
        M.load_original(tmp_path / "a.png")


def test_original_preview_sintetis_sama_dengan_citra_yang_dipakai_mock(tmp_path):
    from launchers.mock import MockLauncher, synthetic_image

    p = RunParams("tiruan_sleman", "cpu", "auto", 3, 2, 42, 1)
    out = MockLauncher(tmp_path / "data").run(p, tmp_path / "run")
    run = read_run(out)
    img, why = M.original_preview(run, tmp_path / "data")
    assert why is None
    np.testing.assert_array_equal(
        img, synthetic_image("tiruan_sleman", run.meta["preview_width"], run.meta["preview_height"])
    )
    assert img.shape[:2] == run.labels_preview.shape


def test_original_preview_dari_png_lepas_dan_pesan_bila_tidak_ada(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(data / "kecil.png")
    run = make_run(tmp_path, "r", dataset="kecil", source_image="kecil.png")
    img, why = M.original_preview(run, data)
    assert why is None and img.shape == (30, 40, 3)
    hilang = make_run(tmp_path, "h", dataset="hilang", source_image="x.png")
    img, why = M.original_preview(hilang, data)
    assert img is None and "tidak ditemukan" in why
