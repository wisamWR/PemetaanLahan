"""Tes program `landcover` (C++, src/cli/landcover.cpp, Langkah 11a) lewat biner hasil `make landcover`.

Dilewati bila biner tidak ada atau tidak bisa dijalankan di sini (mis. di Windows). Jalankan di WSL
setelah `make landcover` (biner: LC_CORE_BIN, bawaan build/landcover):

    .venv/bin/python -m pytest app/tests/test_landcover_cli.py -q

Variabel khusus uji (tidak dipakai aplikasi):
    LC_TEST_CORE_BIN       paksa biner/pembungkus tertentu (mis. skrip yang meneruskan ke WSL)
    LC_TEST_IMAGE_FORMAT   format berkas masukan buatan Pillow, bawaan PNG. Hanya dipakai untuk menguji
                           logika dengan backend pengganti (Netpbm, "PPM"); tes yang khusus PNG dilewati.

Tes ini membandingkan keluaran C++ SECARA EKSAK dengan reference_kmeans.py pada piksel hasil
dekode Pillow, jadi galat urutan baris/kanal atau dekode PNG ikut tertangkap. Perbandingan
antarmode (cpu vs opt/omp/cuda) hanya bermakna setelah kmeans_omp/kmeans_cuda terhubung; selama
hanya ref_kmeans yang tertaut, semua mode memanggil kode yang sama.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import config
from landcover.contract import ContractError, read_run, validate
from landcover.reference_kmeans import init_centroids, kmeans_int

IMAGE_FORMAT = os.getenv("LC_TEST_IMAGE_FORMAT", "PNG")
STAGES = ["baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"]


def _core_path() -> Path | None:
    explicit = os.environ.get("LC_TEST_CORE_BIN")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    if sys.platform == "win32":
        return None
    p = Path(config.CORE_BIN)
    return p if p.is_file() and os.access(p, os.X_OK) else None


@pytest.fixture(scope="module")
def core() -> Path:
    p = _core_path()
    if p is None:
        pytest.skip("biner landcover tidak ada/tidak bisa dijalankan di sini (make landcover; jalankan di WSL)")
    return p


def run_core(core: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run([str(core), *map(str, args)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)


@pytest.fixture(scope="module")
def info(core) -> dict:
    r = run_core(core, "--info")
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def make_image(w: int, h: int, seed: int) -> np.ndarray:
    """Citra terstruktur (blok berwarna + derau) supaya K-Means tidak degenerat."""
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 256, (7, 3))
    idx = rng.integers(0, 7, (h // 16 + 1, w // 16 + 1))
    big = np.kron(idx, np.ones((16, 16), dtype=int))[:h, :w]
    return np.clip(pal[big] + rng.integers(-12, 13, (h, w, 3)), 0, 255).astype(np.uint8)


def save(arr: np.ndarray, path: Path, mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).convert(mode).save(path, format=IMAGE_FORMAT)
    return path


def decoded_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def reference(img: np.ndarray, k: int, iters: int, seed: int):
    px = img.reshape(-1, 3)
    init = init_centroids(px, k, seed)
    labels, centroids, counts = kmeans_int(px, init, iters)
    return init, labels, centroids, counts


def lc_args(src: Path, out: Path, *, mode="cpu", k=5, iters=6, seed=7, extra=()):
    return ["--mode", mode, "--input", src, "--k", k, "--iters", iters, "--seed", seed, "--out", out, *extra]


# --------------------------------------------------------------------------- #

def test_info_adalah_json(info):
    assert info["program"] == "landcover" and info["contract_version"] == 1
    assert set(info["engines"]) == {"cpu", "omp", "cuda"}
    assert isinstance(info["engines"]["omp"], bool) and isinstance(info["engines"]["cuda"], bool)
    assert info["engines"]["cpu"] in ("kmeans_cpu", "ref_kmeans")


def test_run_cpu_cocok_dengan_referensi_eksak(core, tmp_path):
    img = make_image(120, 75, 1)
    src = save(img, tmp_path / "foto.png")
    out = tmp_path / "run"
    r = run_core(core, *lc_args(src, out, extra=["--write-labels"]))
    assert r.returncode == 0, r.stderr
    run = read_run(out)
    assert validate(run) == []
    init, labels, centroids, counts = reference(decoded_rgb(src), 5, 6, 7)
    assert np.array_equal(run.centroids, centroids)
    assert np.array_equal(run.counts, counts)
    assert np.array_equal(np.fromfile(out / "labels.u8", dtype=np.uint8), labels)
    m = run.meta
    assert (m["mock"], m["mode"], m["kmeans_device"], m["threads"], m["dataset"]) == (False, "cpu", "serial", 1, "foto")
    assert (m["width"], m["height"], m["k"], m["iters"], m["seed"]) == (120, 75, 5, 6, 7)
    assert m["init_centroids"] == init.tolist()
    # pratinjau = citra penuh (120x75 < 1024); peta pratinjau = warna centroid
    assert run.labels_preview.shape == (75, 120)
    assert np.array_equal(run.labels_preview, labels.reshape(75, 120))
    assert np.array_equal(run.map_preview, run.centroids[run.labels_preview].astype(np.uint8))
    assert np.array_equal(decoded_rgb(out / "map.png"), centroids.astype(np.uint8)[labels.reshape(75, 120)])
    t = json.loads((out / "timing.json").read_text())
    assert t["unit"] == "ms" and t["total_ms"] > 0 and t["total_ms"] >= t["read_ms"] + t["color_ms"] + t["write_ms"] - 1e-6
    assert list(out.glob("*.tmp")) == []


def test_baris_progres_format_dan_urutan(core, tmp_path):
    src = save(make_image(40, 30, 2), tmp_path / "a.png")
    r = run_core(core, *lc_args(src, tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    stages = [re.fullmatch(r"\[stage\] (\w+) selesai (\d+\.\d\d) s", ln) for ln in r.stdout.splitlines() if ln.startswith("[stage]")]
    assert all(stages), r.stdout
    assert [s.group(1) for s in stages] == STAGES


def test_dataset_berpotongan_sama_dengan_satu_citra(core, tmp_path):
    img = make_image(210, 130, 3)
    src = save(img, tmp_path / "utuh.png")
    folder = tmp_path / "potongan"
    cols, rows, chunks = [0, 70, 150, 210], [0, 60, 130], []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            x0, x1, y0, y1 = cols[j], cols[j + 1], rows[i], rows[i + 1]
            save(img[y0:y1, x0:x1], folder / f"r{i}c{j}.png")
            chunks.append({"file": f"r{i}c{j}.png", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    (folder / "manifest.json").write_text(json.dumps({"width": 210, "height": 130, "pixel_size_m": 10, "chunks": chunks, "source": "uji"}))
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_core(core, *lc_args(src, a, extra=["--write-labels"])).returncode == 0
    r = run_core(core, *lc_args(folder, b, mode="opt", extra=["--threads", 4, "--write-labels", "--name", "utuh"]))
    assert r.returncode == 0, r.stderr
    assert (a / "labels.u8").read_bytes() == (b / "labels.u8").read_bytes()
    ra, rb = read_run(a), read_run(b)
    assert np.array_equal(ra.counts, rb.counts) and np.array_equal(ra.centroids, rb.centroids)
    assert validate(rb) == [] and rb.meta["chunks"] == 6 and rb.meta["source"] == "uji" and rb.meta["pixel_size_m"] == 10


def test_semua_mesin_yang_tersedia_identik_dengan_cpu(core, info, tmp_path):
    """Pemeriksaan `cmp` antarmode (cpu vs opt/omp/cuda). Bermakna penuh bila kmeans_omp/kmeans_cuda terhubung."""
    img = make_image(300, 200, 4)
    src = save(img, tmp_path / "citra.png")
    base = tmp_path / "cpu"
    assert run_core(core, *lc_args(src, base, k=6, extra=["--write-labels"])).returncode == 0
    ref = read_run(base)
    cases = [("auto", None)]
    if info["engines"]["omp"]:
        cases.append(("omp", "omp"))
    if info["engines"]["cuda"]:
        cases.append(("cuda", "cuda"))
    for kmeans, device in cases:
        out = tmp_path / f"opt_{kmeans}"
        r = run_core(core, *lc_args(src, out, mode="opt", k=6, extra=["--kmeans", kmeans, "--threads", 4, "--write-labels"]))
        if kmeans == "cuda" and r.returncode != 0:
            pytest.skip(f"CUDA terhubung tetapi tidak bisa dijalankan di mesin ini: {r.stderr.strip()}")
        assert r.returncode == 0, r.stderr
        run = read_run(out)
        assert validate(run) == []
        if device:
            assert run.meta["kmeans_device"] == device
        assert (base / "labels.u8").read_bytes() == (out / "labels.u8").read_bytes(), f"label beda: kmeans={kmeans}"
        assert np.array_equal(run.centroids, ref.centroids) and np.array_equal(run.counts, ref.counts)


def test_run_ulang_ke_folder_sama_tidak_menyisakan_berkas_basi(core, tmp_path):
    src = save(make_image(64, 48, 5), tmp_path / "x.png")
    out = tmp_path / "run"
    assert run_core(core, *lc_args(src, out, k=4, extra=["--write-labels"])).returncode == 0
    (out / "verification.json").write_text('{"ref": "lain", "identical_pct": 100.0}')
    (out / "sisa.tmp").write_text("sisa")
    (out / "catatan.txt").write_text("dibiarkan")
    assert run_core(core, *lc_args(src, out, k=4)).returncode == 0
    assert not (out / "labels.u8").exists()
    assert not (out / "verification.json").exists()
    assert not (out / "sisa.tmp").exists()
    assert (out / "catatan.txt").is_file()
    assert validate(read_run(out)) == []


def test_run_gagal_di_tengah_tidak_meninggalkan_folder_yang_tampak_valid(core, tmp_path):
    src = save(make_image(64, 48, 6), tmp_path / "x.png")
    out = tmp_path / "run"
    assert run_core(core, *lc_args(src, out, k=3, extra=["--write-labels"])).returncode == 0
    src.write_bytes(src.read_bytes()[:-400])           # header utuh, data terpotong
    r = run_core(core, *lc_args(src, out, k=3, extra=["--write-labels"]))
    assert r.returncode == 1 and r.stderr.strip()
    left = {p.name for p in out.iterdir()}
    assert not left & {"meta.json", "timing.json", "counts.json", "map.png", "labels.u8"}, left
    with pytest.raises(ContractError):
        read_run(out)


def test_manifest_buruk_tidak_menyentuh_keluaran_lama(core, tmp_path):
    img = make_image(64, 48, 8)
    prev = tmp_path / "prev"
    assert run_core(core, *lc_args(save(img, tmp_path / "ok.png"), prev, k=3)).returncode == 0
    before = {p.name: p.read_bytes() for p in prev.iterdir()}
    folder = tmp_path / "celah"
    save(img[:, :32], folder / "a.png")
    (folder / "manifest.json").write_text(json.dumps({"width": 64, "height": 48, "chunks": [{"file": "a.png", "x": 0, "y": 0, "w": 32, "h": 48}]}))
    r = run_core(core, *lc_args(folder, prev, k=3))
    assert r.returncode == 1 and "celah" in r.stderr
    assert {p.name: p.read_bytes() for p in prev.iterdir()} == before


def test_keluaran_tidak_boleh_menimpa_masukan(core, tmp_path):
    src = save(make_image(32, 32, 9), tmp_path / "map.png")
    original = src.read_bytes()
    r = run_core(core, *lc_args(src, tmp_path, k=2))
    assert r.returncode == 1 and "tertimpa" in r.stderr
    assert src.read_bytes() == original


@pytest.mark.parametrize("args, needle", [
    (["--input", "x.png", "--out", "o"], "--mode"),
    (["--mode", "gpu", "--input", "x.png", "--out", "o"], "--mode"),
    (["--mode", "cpu", "--out", "o"], "--input"),
    (["--mode", "cpu", "--input", "x.png"], "--out"),
    (["--mode", "cpu", "--input", "x.png", "--out", "o", "--k", "0"], "--k"),
    (["--mode", "cpu", "--input", "x.png", "--out", "o", "--k", "17"], "--k"),
    (["--mode", "cpu", "--input", "x.png", "--out", "o", "--seed", "4294967296"], "--seed"),
    (["--mode", "cpu", "--input", "x.png", "--out", "o", "--threads", "0"], "--threads"),
    (["--mode", "cpu", "--input", "x.png", "--input", "y.png", "--out", "o"], "belum didukung"),
    (["--mode", "cpu", "--input", "x.png", "--out", "o", "--tidak-ada"], "--tidak-ada"),
])
def test_argumen_salah_keluar_2(core, args, needle):
    r = run_core(core, *args)
    assert r.returncode == 2 and needle in r.stderr, (r.returncode, r.stderr)


def test_masukan_tidak_ada_keluar_1(core, tmp_path):
    r = run_core(core, *lc_args(tmp_path / "tidak_ada.png", tmp_path / "o"))
    assert r.returncode == 1 and "tidak ditemukan" in r.stderr


@pytest.mark.parametrize("engine", ["omp", "cuda"])
def test_mesin_yang_tidak_terhubung_keluar_3(core, info, tmp_path, engine):
    if info["engines"][engine]:
        pytest.skip(f"{engine} terhubung pada build ini")
    src = save(make_image(32, 32, 10), tmp_path / "x.png")
    r = run_core(core, *lc_args(src, tmp_path / "o", mode="opt", extra=["--kmeans", engine]))
    assert r.returncode == 3 and f"kmeans_{engine}" in r.stderr


def test_auto_mencatat_alasan(core, tmp_path):
    src = save(make_image(48, 32, 11), tmp_path / "x.png")
    out = tmp_path / "o"
    assert run_core(core, *lc_args(src, out, mode="opt", extra=["--threads", 3])).returncode == 0
    m = read_run(out).meta
    assert "ambang" in m["kmeans_auto_reason"] and m["kmeans_device"] in ("serial", "omp", "cuda")
    assert m["mode"] == "opt" and m["io_threads"] in (1, 3)


def test_pemanasan_cuda_di_luar_total_dan_hanya_untuk_mesin_cuda(core, info, tmp_path):
    """Kontrak U8: konteks CUDA dipanaskan sebelum pengukuran. Biayanya (ratusan ms) dilaporkan terpisah
    (`warmup_ms`) dan TIDAK masuk `total_ms`; mesin lain tidak dipanaskan."""
    src = save(make_image(160, 120, 13), tmp_path / "x.png")
    cpu_out = tmp_path / "cpu"
    assert run_core(core, *lc_args(src, cpu_out)).returncode == 0
    assert json.loads((cpu_out / "timing.json").read_text())["warmup_ms"] == 0
    assert read_run(cpu_out).meta["cuda_warmup"] is False
    if not info["engines"]["cuda"]:
        pytest.skip("cuda tidak terhubung pada build ini")

    out = tmp_path / "cuda"
    r = run_core(core, *lc_args(src, out, mode="opt", extra=["--kmeans", "cuda"]))
    assert r.returncode == 0, r.stderr
    assert "[info] pemanasan kmeans_cuda:" in r.stdout
    t = json.loads((out / "timing.json").read_text())
    assert t["warmup_ms"] > 0 and read_run(out).meta["cuda_warmup"] is True
    parts = sum(t[k] for k in ("read_ms", "init_ms", "kmeans_ms", "color_ms", "write_ms"))
    # Tanpa pemanasan, total_ms memuat inisialisasi konteks (selisih terhadap jumlah tahap ≈ warmup_ms); dengan
    # pemanasan selisih hanya overhead pemanggilan (beberapa ms). Jadi selisih harus jauh di bawah warmup_ms.
    assert t["total_ms"] >= parts - 1e-6
    assert t["total_ms"] - parts < 0.5 * t["warmup_ms"], (t["total_ms"], parts, t["warmup_ms"])
    assert validate(read_run(out)) == []


@pytest.mark.parametrize("w, h", [(1, 1), (1, 300), (3000, 7)])
def test_ukuran_tepi_valid_dan_cocok_referensi(core, tmp_path, w, h):
    img = make_image(w, h, w + h)
    src = save(img, tmp_path / "t.png")
    out = tmp_path / "o"
    k = 16 if (w, h) == (1, 1) else 4
    assert run_core(core, *lc_args(src, out, k=k, iters=4, seed=4294967295, extra=["--write-labels"])).returncode == 0
    run = read_run(out)
    assert validate(run) == []
    _, labels, centroids, counts = reference(decoded_rgb(src), k, 4, 4294967295)
    assert np.array_equal(run.centroids, centroids) and np.array_equal(run.counts, counts)
    assert np.array_equal(np.fromfile(out / "labels.u8", dtype=np.uint8), labels)


@pytest.mark.skipif(IMAGE_FORMAT != "PNG", reason="khusus PNG: ragam tipe warna")
@pytest.mark.parametrize("mode", ["L", "LA", "RGBA", "P"])
def test_png_ragam_tipe_warna_didekode_seperti_pillow(core, tmp_path, mode):
    src = save(make_image(90, 60, 12), tmp_path / f"{mode}.png", mode=mode)
    out = tmp_path / "o"
    assert run_core(core, *lc_args(src, out, k=4, extra=["--write-labels"])).returncode == 0
    run = read_run(out)
    _, labels, centroids, counts = reference(decoded_rgb(src), 4, 6, 7)
    assert np.array_equal(run.centroids, centroids) and np.array_equal(run.counts, counts)
