"""Tes RunParams/run_id (launchers/base.py) dan MockLauncher (plan Langkah 3).

Semua tulisan ke tmp_path; tidak menyentuh runs/ atau data/ asli."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from landcover.contract import STAGES, read_run, validate
from launchers.base import RunParams, out_dir_for, run_id
from launchers.mock import (
    AUTO_CUDA_THRESHOLD_PIXELS,
    DEFAULT_SIZE,
    MockLauncher,
    resolve_device,
    scale_counts,
    synthetic_image,
)

TILE = 10980
RUN_FILES = {
    "meta.json", "timing.json", "centroids.json", "counts.json",
    "labels_preview.png", "map_preview.png",
}


@pytest.fixture(autouse=True)
def _tanpa_tidur(monkeypatch):
    monkeypatch.delenv("LC_MOCK_SLEEP", raising=False)


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def runs_dir(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def launcher(data_dir):
    return MockLauncher(data_dir, max_side=48)      # K-Means pratinjau kecil: tes cepat


def make_params(**kw) -> RunParams:
    base = dict(dataset="uji", mode="opt", kmeans="auto", k=8, iters=20, seed=42, threads=8)
    base.update(kw)
    return RunParams(**base)


def write_manifest(data_dir: Path, name: str, width=TILE, height=TILE, **extra) -> Path:
    folder = data_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(
        json.dumps({"width": width, "height": height, "pixel_size_m": 10, **extra}), encoding="utf-8"
    )
    return folder


def do_run(launcher, runs_dir, **kw):
    p = make_params(**kw)
    out = launcher.run(p, out_dir_for(runs_dir, "mock", p))
    return p, out, read_run(out)


# --------------------------------------------------------------------------- #
# base.py
# --------------------------------------------------------------------------- #

def test_run_id_contoh_plan():
    p = RunParams("sleman", "opt", "auto", 8, 20, 42, 8)
    assert run_id(p) == "sleman__opt-auto__k8_it20_s42_t8"


def test_run_id_cpu_dinormalkan():
    a = make_params(mode="cpu", kmeans="cuda", threads=16)
    b = make_params(mode="cpu", kmeans="omp", threads=1)
    assert run_id(a) == run_id(b) == "uji__cpu__k8_it20_s42_t1"
    assert run_id(make_params(mode="opt", kmeans="cuda")) != run_id(make_params(mode="opt", kmeans="omp"))
    assert run_id(make_params(threads=4)) != run_id(make_params(threads=8))


def test_out_dir_memisahkan_mock_dan_core(tmp_path):
    p = make_params()
    assert out_dir_for(tmp_path, "mock", p) == tmp_path / "mock" / run_id(p)
    assert out_dir_for(tmp_path, "cli", p) == tmp_path / "core" / run_id(p)
    with pytest.raises(ValueError):
        out_dir_for(tmp_path, "lain", p)


@pytest.mark.parametrize("override", [
    {"dataset": ""}, {"dataset": ".."}, {"dataset": "a/b"}, {"dataset": "a\\b"},
    {"dataset": "C:x"}, {"dataset": " x"}, {"dataset": "x."}, {"dataset": "../x"},
    {"mode": "gpu"}, {"kmeans": "auto2"},
    {"k": 0}, {"k": 17}, {"k": 8.0}, {"k": True},
    {"iters": 0}, {"seed": -1}, {"seed": 2**32}, {"threads": 0},
])
def test_runparams_menolak_nilai_tidak_sah(override):
    with pytest.raises(ValueError):
        make_params(**override)


def test_runparams_menerima_nama_dataset_wajar():
    for name in ("sleman_T49MDM", "kulon-progo", "tile 49.v2", "data1"):
        assert make_params(dataset=name).dataset == name
    assert make_params(seed=2**32 - 1, k=16, threads=1).k == 16


# --------------------------------------------------------------------------- #
# Fungsi murni
# --------------------------------------------------------------------------- #

def test_scale_counts_jumlah_tepat_dan_nol_tetap_nol():
    out = scale_counts(np.array([5, 0, 3]), 1000)
    assert out.tolist() == [625, 0, 375] and out.dtype == np.int64
    # Tidak habis dibagi: [3, 0, 4] ke 100 -> lantai [42, 0, 57] sisa 1 -> cluster terbesar (indeks 2).
    out = scale_counts(np.array([3, 0, 4]), 100)
    assert out.tolist() == [42, 0, 58] and out.sum() == 100
    # Seri untuk terbesar -> indeks terkecil.
    assert scale_counts(np.array([1, 1, 1]), 10).tolist() == [4, 3, 3]
    # Satu cluster berisi semua, lainnya 0.
    assert scale_counts(np.array([0, 7, 0, 0]), TILE * TILE).tolist() == [0, TILE * TILE, 0, 0]


def test_scale_counts_aritmetika_eksak_pada_angka_besar():
    n = 120_560_400
    out = scale_counts(np.array([262144 // 3, 262144 - 262144 // 3, 0]), n)
    assert int(out.sum()) == n and out[2] == 0


@pytest.mark.parametrize("counts,total", [([0, 0], 10), ([-1, 5], 10), ([1, 1], 0)])
def test_scale_counts_menolak_masukan_tidak_sah(counts, total):
    with pytest.raises(ValueError):
        scale_counts(np.array(counts), total)


def test_resolve_device_tidak_pernah_auto():
    big, small = AUTO_CUDA_THRESHOLD_PIXELS, AUTO_CUDA_THRESHOLD_PIXELS - 1
    assert resolve_device(make_params(mode="cpu", kmeans="cuda"), big) == ("serial", None)
    assert resolve_device(make_params(kmeans="omp"), big) == ("omp", None)
    assert resolve_device(make_params(kmeans="cuda"), 1) == ("cuda", None)
    dev, why = resolve_device(make_params(kmeans="auto"), big)
    assert dev == "cuda" and "≥" in why
    dev, why = resolve_device(make_params(kmeans="auto"), small)
    assert dev == "omp" and "<" in why


def test_citra_sintetis_deterministik_dan_hanya_bergantung_nama():
    a = synthetic_image("sleman", 64, 48)
    assert a.shape == (48, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(a, synthetic_image("sleman", 64, 48))
    assert not np.array_equal(a, synthetic_image("kulonprogo", 64, 48))
    # Keempat tutupan (air, vegetasi, tanah, batuan) hadir dengan porsi wajar, bukan satu warna rata.
    big = synthetic_image("sleman", 200, 200).reshape(-1, 3).astype(np.int64)
    palette = np.array([[40, 90, 160], [50, 110, 50], [150, 110, 70], [150, 150, 150]])
    nearest = ((big[:, None, :] - palette[None]) ** 2).sum(axis=2).argmin(axis=1)
    share = np.bincount(nearest, minlength=4) / len(big)
    assert np.allclose(share, [0.15, 0.50, 0.25, 0.10], atol=0.03)
    assert synthetic_image("x", 1, 1).shape == (1, 1, 3)       # ukuran ekstrem tidak error


# --------------------------------------------------------------------------- #
# MockLauncher
# --------------------------------------------------------------------------- #

CASES = [
    ("cpu", "auto", "serial", 1),
    ("opt", "auto", "cuda", 0),       # tile penuh 120,6 juta piksel >= ambang
    ("opt", "omp", "omp", 8),
    ("opt", "cuda", "cuda", 0),
]


@pytest.mark.parametrize("mode,kmeans,device,meta_threads", CASES)
def test_setiap_mode_lolos_validate(launcher, runs_dir, mode, kmeans, device, meta_threads):
    p, out, run = do_run(launcher, runs_dir, mode=mode, kmeans=kmeans)
    assert out == runs_dir / "mock" / run_id(p)
    assert {f.name for f in out.iterdir()} == RUN_FILES            # tanpa sisa .tmp
    assert validate(run) == []
    assert run.mock is True
    assert run.meta["kmeans_device"] == device and run.meta["threads"] == meta_threads
    assert run.meta["mode"] == mode and run.mode == mode
    assert (run.width, run.height) == DEFAULT_SIZE and run.n_pixels == TILE * TILE
    assert run.stages["kmeans"].device == device
    io = "serial" if mode == "cpu" else "omp"
    assert [run.stages[s].device for s in STAGES] == [io, "serial", device, io, io]
    # h2d/kernel/d2h hanya untuk CUDA (kontrak: selain itu 0).
    t = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert t["unit"] == "ms" and set(t["devices"]) == {"read", "init", "kmeans", "color", "write"}
    if device == "cuda":
        assert run.stages["kmeans"].detail is not None and run.stages["kmeans"].detail["kernel"] > 0
        assert t["kmeans_ms"] >= t["h2d_ms"] + t["kernel_ms"] + t["d2h_ms"]
    else:
        assert run.stages["kmeans"].detail is None
        assert t["h2d_ms"] == t["kernel_ms"] == t["d2h_ms"] == 0
    assert (kmeans == "auto" and mode == "opt") == ("kmeans_auto_reason" in run.meta)


def test_cpu_dan_opt_label_identik_tetapi_cpu_lebih_lambat(launcher, runs_dir):
    _, _, cpu = do_run(launcher, runs_dir, mode="cpu")
    for kmeans in ("auto", "omp", "cuda"):
        _, _, opt = do_run(launcher, runs_dir, mode="opt", kmeans=kmeans, threads=8)
        assert np.array_equal(cpu.labels_preview, opt.labels_preview)
        assert np.array_equal(cpu.map_preview, opt.map_preview)
        assert np.array_equal(cpu.centroids, opt.centroids)
        assert np.array_equal(cpu.counts, opt.counts)
        assert cpu.total > opt.total
        assert cpu.stages["kmeans"].time > opt.stages["kmeans"].time
        # jauh di atas jitter +-5 %, dan satuan waktu masuk akal (bukan ms yang tertukar detik)
        assert cpu.total / opt.total > 2
    assert 1.0 < cpu.total < 1000.0


def test_hasil_tidak_bergantung_jumlah_thread(launcher, runs_dir):
    _, _, a = do_run(launcher, runs_dir, kmeans="omp", threads=2)
    _, _, b = do_run(launcher, runs_dir, kmeans="omp", threads=16)
    assert np.array_equal(a.labels_preview, b.labels_preview)
    assert np.array_equal(a.counts, b.counts)
    assert a.total > b.total                      # lebih banyak thread -> lebih cepat
    # baca/pewarnaan/tulis memakai OpenMP pada mode opt, jadi ikut mengecil (bukan hanya K-Means)
    for stage in ("baca", "kmeans", "pewarnaan", "tulis"):
        assert a.stages[stage].time > b.stages[stage].time, stage


def test_mode_cpu_mengabaikan_pilihan_kmeans(launcher, runs_dir):
    _, out_a, a = do_run(launcher, runs_dir, mode="cpu", kmeans="cuda")
    _, out_b, b = do_run(launcher, runs_dir, mode="cpu", kmeans="omp")
    assert out_a == out_b and a.meta["kmeans_device"] == b.meta["kmeans_device"] == "serial"


def test_dua_kali_jalan_hasilnya_sama(launcher, runs_dir, tmp_path):
    p = make_params(mode="opt", kmeans="cuda")
    one = launcher.run(p, tmp_path / "a")
    two = launcher.run(p, tmp_path / "b")
    for name in RUN_FILES - {"meta.json"}:
        assert (one / name).read_bytes() == (two / name).read_bytes(), name
    meta1 = json.loads((one / "meta.json").read_text(encoding="utf-8"))
    meta2 = json.loads((two / "meta.json").read_text(encoding="utf-8"))
    meta1.pop("created_at"), meta2.pop("created_at")
    assert meta1 == meta2


def test_jalan_ulang_ke_folder_yang_sama_menimpa_bersih(launcher, tmp_path):
    out = tmp_path / "r"
    launcher.run(make_params(k=4), out)
    launcher.run(make_params(k=6), out)
    assert {f.name for f in out.iterdir()} == RUN_FILES
    assert read_run(out).k == 6 and validate(read_run(out)) == []


def test_jalan_ulang_membuang_berkas_opsional_basi(launcher, tmp_path):
    """labels.u8/verification.json/map.png dari run lama bukan milik run baru: verifikasi dan ekspor
    penuh akan membacanya sebagai milik run baru bila dibiarkan."""
    out = tmp_path / "r"
    launcher.run(make_params(k=4), out)
    for name in ("labels.u8", "verification.json", "map.png"):
        (out / name).write_bytes(b"basi")
    (out / "catatan.txt").write_text("bukan berkas run", encoding="utf-8")
    launcher.run(make_params(k=4), out)
    assert {f.name for f in out.iterdir()} == RUN_FILES | {"catatan.txt"}


def test_k_berbeda_memberi_k_centroid_dan_counts(launcher, runs_dir):
    for k in (1, 2, 5, 16):
        _, _, run = do_run(launcher, runs_dir, k=k)
        assert len(run.centroids) == len(run.counts) == k
        assert validate(run) == [] and int(run.labels_preview.max()) < k


def test_ukuran_dari_manifest_dan_pratinjau_dari_preview_jpg(launcher, data_dir, runs_dir):
    folder = write_manifest(data_dir, "wilayah", width=TILE, height=TILE // 2, pixel_size_m=20)
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)).save(folder / "preview.jpg")
    _, _, run = do_run(launcher, runs_dir, dataset="wilayah")
    assert (run.width, run.height) == (TILE, TILE // 2)
    assert run.meta["pixel_size_m"] == 20 and run.meta["source_image"] == "preview.jpg"
    assert (run.meta["preview_width"], run.meta["preview_height"]) == (48, 24)
    assert run.labels_preview.shape == (24, 48) == run.map_preview.shape[:2]
    assert validate(run) == []


def test_citra_sintetis_sama_antar_proses_python(tmp_path):
    # hash() str diacak per proses (PYTHONHASHSEED); hasil mock harus tetap sama setelah
    # aplikasi dijalankan ulang, jadi seed citra memakai crc32. Hanya subproses yang bisa
    # menangkap kesalahan ini.
    import hashlib
    import os
    import subprocess
    import sys

    app_dir = Path(__file__).resolve().parent.parent
    kode = (
        "import hashlib, sys\n"
        "from launchers.mock import synthetic_image\n"
        "sys.stdout.write(hashlib.sha256(synthetic_image('sleman', 64, 48).tobytes()).hexdigest())\n"
    )
    ini = hashlib.sha256(synthetic_image("sleman", 64, 48).tobytes()).hexdigest()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(app_dir)}
        hasil = subprocess.run(
            [sys.executable, "-c", kode], env=env, cwd=tmp_path,
            capture_output=True, text=True, check=True,
        )
        assert hasil.stdout == ini


def test_dataset_satu_png_tanpa_manifest_dan_cluster_kosong(launcher, data_dir, runs_dir):
    # Citra satu warna, K = 4: tiga cluster kosong (kontrak U2) tetap valid dan counts = 0.
    folder = data_dir / "seragam"
    folder.mkdir()
    Image.fromarray(np.full((8, 8, 3), (30, 90, 40), dtype=np.uint8)).save(folder / "tile.png")
    _, _, run = do_run(launcher, runs_dir, dataset="seragam", k=4)
    assert (run.width, run.height) == (8, 8) and run.meta["source_image"] == "tile.png"
    assert run.counts.tolist() == [64, 0, 0, 0]
    assert (run.labels_preview == 0).all() and validate(run) == []


def test_tanpa_folder_dataset_memakai_citra_sintetis_ukuran_tile(launcher, runs_dir):
    _, _, run = do_run(launcher, runs_dir, dataset="tidak_ada")
    assert run.meta["source_image"] == "synthetic"
    assert (run.width, run.height) == DEFAULT_SIZE and run.meta["pixel_size_m"] == 10
    assert validate(run) == []


def test_manifest_rusak_ditolak_jelas(launcher, data_dir, tmp_path):
    folder = data_dir / "rusak"
    folder.mkdir()
    for isi in ("{bukan json", '{"width": 0, "height": 5}', '{"width": 5}', "[1, 2]"):
        (folder / "manifest.json").write_text(isi, encoding="utf-8")
        with pytest.raises(ValueError, match="manifest.json"):
            launcher.run(make_params(dataset="rusak"), tmp_path / "x")


def test_break_verification_hanya_mengubah_run_opt(data_dir, runs_dir, launcher):
    broken = MockLauncher(data_dir, max_side=48, break_verification=True)
    _, _, normal = do_run(launcher, runs_dir / "n", mode="opt", kmeans="cuda")
    _, _, bad = do_run(broken, runs_dir / "b", mode="opt", kmeans="cuda")
    _, _, cpu_ok = do_run(launcher, runs_dir / "n", mode="cpu")
    _, _, cpu_b = do_run(broken, runs_dir / "b", mode="cpu")

    assert validate(bad) == []                                  # tetap kontrak-valid
    diff = int((normal.labels_preview != bad.labels_preview).sum())
    assert 1 <= diff <= normal.labels_preview.size // 50        # ~1 %, bukan kekacauan
    assert not np.array_equal(normal.counts, bad.counts)
    assert np.array_equal(normal.centroids, bad.centroids)
    # peta mengikuti label yang diubah
    assert np.array_equal(bad.map_preview, bad.centroids[bad.labels_preview].astype(np.uint8))
    # run cpu (acuan) tidak disentuh, jadi cpu vs opt sengaja berbeda
    assert np.array_equal(cpu_ok.labels_preview, cpu_b.labels_preview)
    assert not np.array_equal(cpu_ok.labels_preview, bad.labels_preview)
    # deterministik
    _, _, bad2 = do_run(broken, runs_dir / "b2", mode="opt", kmeans="cuda")
    assert np.array_equal(bad.labels_preview, bad2.labels_preview)


def test_break_verification_menolak_k_1(data_dir, runs_dir):
    broken = MockLauncher(data_dir, max_side=48, break_verification=True)
    with pytest.raises(ValueError, match="k >= 2"):
        do_run(broken, runs_dir, mode="opt", k=1)
    do_run(broken, runs_dir, mode="cpu", k=1)                   # cpu tidak terpengaruh


def test_on_progress_melaporkan_lima_tahap_berurutan(launcher, tmp_path):
    lines: list[str] = []
    launcher.run(make_params(), tmp_path / "r", on_progress=lines.append)
    stage_lines = [x for x in lines if x.startswith("[stage]")]
    assert [x.split()[1] for x in stage_lines] == list(STAGES)
    assert all(x.endswith(" s") and "selesai" in x for x in stage_lines)
    assert any("TIRUAN" in x for x in lines)


def test_tidur_dari_env_lc_mock_sleep(data_dir, tmp_path, monkeypatch):
    sleeps: list[float] = []
    mk = lambda **kw: MockLauncher(data_dir, max_side=16, sleep_fn=sleeps.append, **kw)

    mk().run(make_params(), tmp_path / "a")
    assert sleeps == []                                          # env tidak diset: tanpa tidur

    monkeypatch.setenv("LC_MOCK_SLEEP", "1")
    mk().run(make_params(), tmp_path / "b")
    assert len(sleeps) == 5 and sum(sleeps) == pytest.approx(1.0)

    sleeps.clear()
    mk(sleep=0.0).run(make_params(), tmp_path / "c")             # argumen menang atas env
    assert sleeps == []

    monkeypatch.setenv("LC_MOCK_SLEEP", "0")
    mk().run(make_params(), tmp_path / "d")
    assert sleeps == []

    for buruk in ("abc", "-1", "nan"):
        monkeypatch.setenv("LC_MOCK_SLEEP", buruk)
        with pytest.raises(ValueError, match="LC_MOCK_SLEEP"):
            mk().run(make_params(), tmp_path / "e")


def test_data_dir_bawaan_dari_config(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "dari_config")
    assert MockLauncher().data_dir == tmp_path / "dari_config"


def test_argumen_konstruktor_tidak_sah():
    with pytest.raises(ValueError):
        MockLauncher(max_side=0)


def test_params_dipakai_apa_adanya_bukan_dimodifikasi(launcher, tmp_path):
    p = make_params()
    before = replace(p)
    launcher.run(p, tmp_path / "r")
    assert p == before


def test_nodata_tepi_hitam_pada_mock_launcher(data_dir, runs_dir):
    # Citra kecil <= max_side (tanpa resize): nodata_pixels dan nodata_cluster persis.
    img = np.full((32, 32, 3), (40, 140, 60), dtype=np.uint8)
    img[:4, :] = 0
    img[:, :4] = 0
    expected_nd = int(((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)).sum())
    Image.fromarray(img).save(data_dir / "tepi_hitam.png")

    launcher = MockLauncher(data_dir, max_side=512)
    _, _, run = do_run(launcher, runs_dir, dataset="tepi_hitam", k=3)
    assert validate(run) == []
    assert run.nodata_pixels == expected_nd
    assert 0 <= run.nodata_cluster < 3 and run.counts[run.nodata_cluster] == expected_nd

    # break_verification pada citra ber-nodata tetap lolos validate()
    broken = MockLauncher(data_dir, max_side=512, break_verification=True)
    _, _, bad = do_run(broken, runs_dir / "brk", dataset="tepi_hitam", k=3)
    assert validate(bad) == []
