"""Tes CliLauncher (plan Langkah 11) dengan skrip Python palsu `fake_core.py` sebagai pengganti biner.

Skrip palsu meniru `landcover` pada PNG kecil (K-Means referensi, folder run kontrak 4.3) dan bisa
dipaksa gagal/rusak/menggantung lewat variabel lingkungan (lihat kepala `fake_core.py`). Tes dengan
biner NYATA ada di `test_cli_launcher_real.py` (dilewati bila biner tidak ada).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from conftest import write_run_dir
from landcover.contract import read_run, validate
from landcover.verify import FULL_SOURCES, STATUS_IDENTICAL, verify_pair
from launchers import cli
from launchers.base import RunParams, out_dir_for, run_id
from launchers.cli import (
    CliLauncher, CliLauncherError, CoreFailedError, CoreInfo, CoreNotFoundError, CoreOutputError,
    CoreTimeoutError, build_command, clear_run_files, meta_mismatches, parse_core_info, probe_core,
    resolve_timeout,
)

FAKE = Path(__file__).resolve().parent / "fake_core.py"
APP_FILE = str(config.ROOT / "app" / "streamlit_app.py")
posix_only = pytest.mark.skipif(os.name != "posix", reason="memakai sinyal POSIX / bit eksekusi")


def fake_cmd() -> list[str]:
    return [sys.executable, str(FAKE)]


def P(**kw) -> RunParams:
    base = dict(dataset="kecil", mode="cpu", kmeans="auto", k=4, iters=3, seed=7, threads=4)
    base.update(kw)
    return RunParams(**base)


@pytest.fixture(autouse=True)
def bersih(monkeypatch):
    for name in ("FAKE_CORE", "FAKE_CORE_DROP", "FAKE_ENGINES", "FAKE_CORE_LOG", cli.TIMEOUT_ENV):
        monkeypatch.delenv(name, raising=False)
    cli._PROBE_CACHE.clear()
    yield
    cli._PROBE_CACHE.clear()


@pytest.fixture
def data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(d / "kecil.png")
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(d / "peta sleman.png")
    tile = d / "tile"
    tile.mkdir()
    img = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
    Image.fromarray(img[:, :32]).save(tile / "r0c0.png")
    Image.fromarray(img[:, 32:]).save(tile / "r0c1.png")
    (tile / "manifest.json").write_text(json.dumps({
        "width": 64, "height": 32, "pixel_size_m": 10,
        "chunks": [{"file": "r0c0.png", "x": 0, "y": 0, "w": 32, "h": 32},
                   {"file": "r0c1.png", "x": 32, "y": 0, "w": 32, "h": 32}],
    }), encoding="utf-8")
    return d


@pytest.fixture
def runs(tmp_path) -> Path:
    return tmp_path / "runs"


def launcher(data_dir, *, core=None, **kw) -> CliLauncher:
    kw.setdefault("timeout", 60)
    return CliLauncher(data_dir, fake_cmd() if core is None else core, **kw)


def go(data_dir, runs, params=None, *, core=None, **kw):
    """Jalankan satu run; (folder keluaran, baris progres)."""
    params = params or P()
    out = out_dir_for(runs, "cli", params)
    lines: list[str] = []
    result = launcher(data_dir, core=core, **kw).run(params, out, lines.append)
    assert result == out
    return out, lines


def old_run(out: Path) -> None:
    """Isi `out` dengan run lama yang lengkap (mock) + berkas basi + sisa .tmp + berkas lain."""
    write_run_dir(out, width=48, height=32, k=4)
    for name in ("map.png", "labels.u8", "verification.json"):
        (out / name).write_bytes(b"basi")
    (out / "sisa.json.tmp").write_bytes(b"x")
    (out / "catatan.txt").write_text("jangan dihapus", encoding="utf-8")


# --- build_command ------------------------------------------------------------- #

def test_build_command_cpu_tanpa_kmeans_dan_threads(tmp_path):
    cmd = build_command(P(mode="cpu", kmeans="cuda", threads=16), tmp_path / "out", tmp_path / "x.png", core=["lc"])
    assert cmd == ["lc", "--mode", "cpu", "--input", str(tmp_path / "x.png"), "--name", "kecil",
                   "--k", "4", "--iters", "3", "--seed", "7", "--out", str(tmp_path / "out"), "--write-labels"]


@pytest.mark.parametrize("kmeans", ["auto", "omp", "cuda"])
def test_build_command_opt(tmp_path, kmeans):
    cmd = build_command(P(mode="opt", kmeans=kmeans, threads=6, k=8, iters=20, seed=42),
                        "out", "d/x.png", core=["lc"])
    assert cmd == ["lc", "--mode", "opt", "--kmeans", kmeans, "--input", "d/x.png", "--name", "kecil",
                   "--k", "8", "--iters", "20", "--seed", "42", "--threads", "6", "--out", "out",
                   "--write-labels"]


def test_build_command_write_labels_bisa_dimatikan_dan_tiap_nilai_satu_argumen(tmp_path):
    cmd = build_command(P(dataset="peta sleman"), tmp_path / "ada spasi", tmp_path / "x.png",
                        core=["lc"], write_labels=False)
    assert "--write-labels" not in cmd
    assert cmd[cmd.index("--name") + 1] == "peta sleman"                  # satu argumen, spasi utuh
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "ada spasi")
    assert not any("=" in a for a in cmd if a.startswith("--"))           # tanpa --flag=nilai (pembungkus WSL)


def test_command_memakai_path_hasil_resolve_dataset(data_dir, runs):
    lc = launcher(data_dir)
    png = lc.command(P(), out_dir_for(runs, "cli", P()))
    assert png[png.index("--input") + 1] == str(data_dir / "kecil.png")
    folder = lc.command(P(dataset="tile"), out_dir_for(runs, "cli", P(dataset="tile")))
    assert folder[folder.index("--input") + 1] == str(data_dir / "tile")
    assert folder[folder.index("--name") + 1] == "tile"


@pytest.mark.parametrize("name", ["tidak_ada", "tiruan_sleman"])       # tidak ada, dan sintetis (hanya mock)
def test_dataset_tidak_ditemukan_valueerror_sebelum_proses_dimulai(data_dir, runs, name):
    out = out_dir_for(runs, "cli", P(dataset=name))
    with pytest.raises(ValueError, match="tidak ditemukan"):
        launcher(data_dir).run(P(dataset=name), out)
    assert not out.exists()


def test_folder_keluaran_tidak_boleh_bertumpuk_dengan_masukan(data_dir):
    lc = launcher(data_dir)
    with pytest.raises(ValueError, match="bertumpuk"):
        lc.command(P(dataset="tile"), data_dir / "tile")
    with pytest.raises(ValueError, match="bertumpuk"):
        lc.command(P(dataset="tile"), data_dir / "tile" / "run")
    with pytest.raises(ValueError, match="bertumpuk"):
        lc.command(P(), data_dir)                                      # PNG lepas: map.png akan di data/
    lc.command(P(), data_dir / "runs" / "satu")                        # folder lain di bawah data/ boleh


# --- jalan sukses --------------------------------------------------------------- #

def test_run_cpu_menghasilkan_folder_valid_dan_meneruskan_baris_progres(data_dir, runs):
    out, lines = go(data_dir, runs)
    assert out == runs / "core" / "kecil__cpu__k4_it3_s7_t1"
    run = read_run(out)
    assert validate(run) == []
    m = run.meta
    assert (m["mock"], m["dataset"], m["mode"], m["k"], m["iters"], m["seed"]) == (False, "kecil", "cpu", 4, 3, 7)
    assert (m["width"], m["height"]) == (48, 32) and m["kmeans_device"] == "serial"
    assert (out / "labels.u8").stat().st_size == 48 * 32                    # --write-labels selalu
    stages = [ln for ln in lines if ln.startswith("[stage]")]
    assert [s.split()[1] for s in stages] == ["baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"]
    assert lines[0].startswith("[info] dataset 'kecil'") and lines[-1].startswith("[info] folder run:")


def test_run_opt_meneruskan_peringatan_dan_threads(data_dir, runs, monkeypatch):
    log = runs.parent / "log.jsonl"
    monkeypatch.setenv("FAKE_CORE_LOG", str(log))
    p = P(mode="opt", kmeans="auto", threads=3)
    out, lines = go(data_dir, runs, p)
    assert validate(read_run(out)) == []
    assert any(ln.startswith("[peringatan]") for ln in lines)
    argv = json.loads(log.read_text(encoding="utf-8").splitlines()[0])["argv"]
    assert argv[argv.index("--mode") + 1] == "opt" and argv[argv.index("--threads") + 1] == "3"
    assert argv[argv.index("--kmeans") + 1] == "auto"


def test_run_dataset_folder_dengan_potongan(data_dir, runs):
    out, _ = go(data_dir, runs, P(dataset="tile"))
    run = read_run(out)
    assert validate(run) == [] and (run.meta["width"], run.meta["height"], run.meta["chunks"]) == (64, 32, 2)


def test_run_tanpa_callback_dan_nama_dataset_dengan_spasi(data_dir, runs):
    p = P(dataset="peta sleman")
    out = out_dir_for(runs, "cli", p)
    launcher(data_dir).run(p, out)                                          # on_progress = None
    assert read_run(out).meta["dataset"] == "peta sleman"


def test_baris_utf8_utuh_dan_semua_baris_tiba_sebelum_run_kembali(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "utf8")
    _, lines = go(data_dir, runs)
    assert "[info] auto: N = 0,0 juta piksel ≥ ambang 8 juta" in lines
    assert lines[-1].startswith("[info] folder run:")                       # baris terakhir sudah diterima


def test_semua_baris_diterima_walau_callback_lambat_dan_proses_sudah_selesai(data_dir, runs):
    lines: list[str] = []

    def lambat(line: str) -> None:
        time.sleep(0.05)                      # proses sudah keluar jauh sebelum baris terakhir diproses
        lines.append(line)

    launcher(data_dir).run(P(), out_dir_for(runs, "cli", P()), lambat)
    assert [ln.split()[1] for ln in lines if ln.startswith("[stage]")] == \
        ["baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"]
    assert lines[-1].startswith("[info] folder run:")


def test_baris_rusak_dibersihkan_dan_dipotong(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "garbage")
    lines: list[str] = []
    with pytest.raises(CoreOutputError):
        launcher(data_dir).run(P(), out_dir_for(runs, "cli", P()), lines.append)
    assert "keluaran rusak" in lines[0] and len(lines[0]) <= cli.LINE_CHARS
    assert not any(ord(c) < 32 or ord(c) == 127 for ln in lines for c in ln)
    assert "[stage] baca selesai tidak-angka" in lines                      # diteruskan apa adanya


# --- pembersihan berkas basi ------------------------------------------------------ #

def test_berkas_basi_dibuang_sebelum_biner_dipanggil(data_dir, runs, monkeypatch):
    p = P()
    out = out_dir_for(runs, "cli", p)
    old_run(out)
    log = runs.parent / "log.jsonl"
    monkeypatch.setenv("FAKE_CORE_LOG", str(log))
    monkeypatch.setenv("FAKE_CORE", "staleguard")             # keluar 9 bila ada berkas run lama
    launcher(data_dir).run(p, out)
    seen = json.loads(log.read_text(encoding="utf-8").splitlines()[0])["out_files_at_start"]
    assert seen == ["catatan.txt"]                                          # berkas lain dibiarkan
    assert (out / "catatan.txt").read_text(encoding="utf-8") == "jangan dihapus"
    assert validate(read_run(out)) == []


def test_labels_u8_basi_hilang_bila_write_labels_mati(data_dir, runs):
    p = P()
    out = out_dir_for(runs, "cli", p)
    old_run(out)
    launcher(data_dir, write_labels=False).run(p, out)
    assert not (out / "labels.u8").exists() and not (out / "verification.json").exists()
    assert not list(out.glob("*.tmp"))


def test_biner_keluar_0_tanpa_menulis_tidak_mewarisi_run_lama(data_dir, runs, monkeypatch):
    p = P()
    out = out_dir_for(runs, "cli", p)
    old_run(out)
    monkeypatch.setenv("FAKE_CORE", "silent0")
    with pytest.raises(CoreOutputError, match="tidak terbaca") as ei:
        launcher(data_dir).run(p, out)
    assert "meta.json" in ei.value.detail                                  # run lama tidak tampil sebagai baru


def test_run_gagal_meninggalkan_folder_tanpa_meta(data_dir, runs, monkeypatch):
    p = P()
    out = out_dir_for(runs, "cli", p)
    old_run(out)
    monkeypatch.setenv("FAKE_CORE", "exit1")
    with pytest.raises(CoreFailedError):
        launcher(data_dir).run(p, out)
    assert not (out / "meta.json").exists()                                # tidak pernah tampak valid


def test_clear_run_files_hanya_berkas_biasa(tmp_path):
    clear_run_files(tmp_path / "tidak_ada")                                # folder tidak ada: tidak apa-apa
    out = tmp_path / "o"
    out.mkdir()
    (out / "meta.json").write_text("{}")
    (out / "map.png").mkdir()                                              # direktori bernama map.png: dibiarkan
    (out / "a.tmp").write_text("x")
    (out / "lain.txt").write_text("x")
    clear_run_files(out)
    assert sorted(p.name for p in out.iterdir()) == ["lain.txt", "map.png"]
    assert cli.RUN_FILES[0] == "meta.json"                                 # penanda run lengkap dihapus duluan


# --- kegagalan biner -------------------------------------------------------------- #

@pytest.mark.parametrize("mode, code, summary, stderr", [
    ("exit1", 1, "gagal saat berjalan", "masukan rusak"),
    ("exit2", 2, "menolak argumen", "harus bilangan bulat"),
    ("exit3", 3, "tidak terhubung", "kmeans_cuda tidak terhubung"),
])
def test_kode_keluar_1_2_3_menampilkan_stderr(data_dir, runs, monkeypatch, mode, code, summary, stderr):
    monkeypatch.setenv("FAKE_CORE", mode)
    with pytest.raises(CoreFailedError) as ei:
        launcher(data_dir).run(P(), out_dir_for(runs, "cli", P()))
    e = ei.value
    assert isinstance(e, CliLauncherError) and e.exit_code == code
    assert summary in e.summary and stderr in e.detail and stderr in str(e)


def test_kode_tak_dikenal_dan_stderr_kosong_memakai_ekor_stdout(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "exit7")
    with pytest.raises(CoreFailedError, match="kode 7 .tidak dikenal") as ei:
        launcher(data_dir).run(P(), out_dir_for(runs, "cli", P()))
    assert ei.value.exit_code == 7 and ei.value.detail == ""


@posix_only
def test_proses_dimatikan_sinyal(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "signal")
    with pytest.raises(CoreFailedError, match="sinyal 9") as ei:
        launcher(data_dir).run(P(), out_dir_for(runs, "cli", P()))
    assert ei.value.exit_code == -9


def test_stderr_besar_tidak_membuat_deadlock_dan_ekornya_yang_tampil(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "bigerr")
    lines: list[str] = []
    t0 = time.monotonic()
    with pytest.raises(CoreFailedError) as ei:
        launcher(data_dir, timeout=30).run(P(), out_dir_for(runs, "cli", P()), lines.append)
    assert time.monotonic() - t0 < 25
    d = ei.value.detail
    assert d.endswith("akhir stderr panjang") and len(d) <= cli.DETAIL_CHARS and d.startswith("…")
    assert len([ln for ln in lines if ln.startswith("[info] langkah")]) == 11


def test_batas_waktu_mematikan_proses_dan_cepat(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "hang")                                 # tidur 30 detik
    lines: list[str] = []
    t0 = time.monotonic()
    with pytest.raises(CoreTimeoutError, match="batas waktu 1 s") as ei:
        launcher(data_dir, timeout=1.0).run(P(), out_dir_for(runs, "cli", P()), lines.append)
    assert time.monotonic() - t0 < 20
    assert lines == ["[info] mulai"] and "LC_CORE_TIMEOUT" in ei.value.summary


def test_biner_tidak_ada_pesan_jelas_dan_folder_lama_tidak_disentuh(data_dir, runs, tmp_path):
    p = P()
    out = out_dir_for(runs, "cli", p)
    old_run(out)
    with pytest.raises(CoreNotFoundError, match="make landcover") as ei:
        launcher(data_dir, core=tmp_path / "tidak" / "ada" / "landcover").run(p, out)
    assert "tidak ditemukan" in ei.value.summary
    assert validate(read_run(out)) == []                                    # belum dibersihkan


def test_nama_program_bukan_path_dicari_di_path(data_dir, runs):
    with pytest.raises(CoreNotFoundError, match="di PATH"):
        launcher(data_dir, core="landcover-tidak-ada-xyz").run(P(), out_dir_for(runs, "cli", P()))


@posix_only
def test_berkas_tanpa_izin_eksekusi(data_dir, runs, tmp_path):
    f = tmp_path / "landcover"
    f.write_text("#!/bin/sh\n")
    f.chmod(0o644)
    with pytest.raises(CoreNotFoundError, match="izin eksekusi"):
        launcher(data_dir, core=f).run(P(), out_dir_for(runs, "cli", P()))


def test_berkas_yang_bukan_program_gagal_dimulai(data_dir, runs, tmp_path):
    f = tmp_path / ("rusak.exe" if os.name == "nt" else "rusak")
    f.write_text("ini bukan program")
    f.chmod(0o755)
    with pytest.raises(CoreNotFoundError, match="tidak bisa dijalankan"):
        launcher(data_dir, core=f).run(P(), out_dir_for(runs, "cli", P()))


def test_galat_callback_tidak_membuat_proses_menggantung(data_dir, runs, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "bigerr")

    def boom(_line):
        raise RuntimeError("callback rusak")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="callback rusak"):
        launcher(data_dir, timeout=30).run(P(), out_dir_for(runs, "cli", P()), boom)
    assert time.monotonic() - t0 < 25


# --- keluaran rusak setelah kode 0 -------------------------------------------------- #

@pytest.mark.parametrize("env, expect", [
    ({"FAKE_CORE": "badjson"}, "counts.json"),
    ({"FAKE_CORE_DROP": "map_preview.png"}, "map_preview.png"),
    ({"FAKE_CORE_DROP": "meta.json"}, "meta.json"),
    ({"FAKE_CORE": "wrongname"}, "meta.dataset = 'lain'"),
    ({"FAKE_CORE": "mock"}, "meta.mock"),
])
def test_keluaran_rusak_menjadi_coreoutputerror_dengan_daftar_masalah(data_dir, runs, monkeypatch, env, expect):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(CoreOutputError) as ei:
        launcher(data_dir).run(P(), out_dir_for(runs, "cli", P()))
    assert expect in ei.value.detail and ei.value.exit_code == 0 and "kode 0" in ei.value.summary


def run_with_meta(tmp_path, **meta):
    base = {"mock": False, "dataset": "kecil", "mode": "cpu", "kmeans_device": "serial",
            "k": 4, "iters": 3, "seed": 7}
    base.update(meta)
    return read_run(write_run_dir(tmp_path / f"r{abs(hash(json.dumps(meta, sort_keys=True)))}", k=4, meta=base))


def test_meta_mismatches(tmp_path):
    assert meta_mismatches(run_with_meta(tmp_path), P()) == []
    assert meta_mismatches(run_with_meta(tmp_path, kmeans_device="cuda", mode="opt"), P(mode="opt", kmeans="cuda")) == []
    assert any("mock" in s for s in meta_mismatches(run_with_meta(tmp_path, mock=True), P()))
    assert any("meta.k" in s for s in meta_mismatches(run_with_meta(tmp_path, k=4), P(k=5)))
    assert any("serial" in s for s in meta_mismatches(run_with_meta(tmp_path, kmeans_device="omp"), P(mode="cpu")))
    # mesin yang diminta eksplisit harus yang dipakai; auto boleh apa saja
    omp = run_with_meta(tmp_path, mode="opt", kmeans_device="omp")
    assert any("diminta 'cuda'" in s for s in meta_mismatches(omp, P(mode="opt", kmeans="cuda")))
    assert meta_mismatches(omp, P(mode="opt", kmeans="auto")) == []


# --- kemampuan build (--info) --------------------------------------------------------- #

def info_text(**over) -> str:
    d = {"program": "landcover", "contract_version": 1, "openmp": True, "max_threads": 20,
         "engines": {"cpu": "ref_kmeans", "omp": False, "cuda": False}, "auto_cuda_threshold_pixels": 8000000}
    d.update(over)
    return json.dumps(d)


def test_parse_core_info():
    i = parse_core_info(info_text())
    assert i == CoreInfo("ref_kmeans", False, False, True, 20, 8000000)
    assert parse_core_info(info_text(engines={"cpu": None, "omp": True, "cuda": True})).cpu is None


@pytest.mark.parametrize("text", [
    "bukan json", "[]", "{}", info_text(engines=[]),
    info_text(engines={"cpu": 3, "omp": True, "cuda": True}),
    info_text(engines={"cpu": "x", "omp": "ya", "cuda": True}),
    info_text(engines={"cpu": "x", "omp": True}),
    info_text(max_threads=0), info_text(max_threads=True), info_text(auto_cuda_threshold_pixels="8"),
])
def test_parse_core_info_menolak_bentuk_tak_dikenal(text):
    with pytest.raises(ValueError):
        parse_core_info(text)


def test_pilihan_kmeans_dan_catatan_per_kombinasi_mesin():
    def info(omp, cuda, cpu="ref_kmeans"):
        return CoreInfo(cpu, omp, cuda)

    assert info(False, False).kmeans_choices == ("auto",)
    assert info(True, False).kmeans_choices == ("auto", "omp")
    assert info(False, True).kmeans_choices == ("auto", "cuda")
    assert info(True, True).kmeans_choices == ("auto", "omp", "cuda")
    assert "serial" in info(False, False).notice and "OpenMP dan CUDA" in info(False, False).notice
    assert "CUDA tidak terhubung" in info(True, False).notice
    assert "kmeans_omp" in info(False, True).notice
    assert info(True, True).notice is None
    assert "tidak punya mesin CPU" in info(True, True, cpu=None).notice


def test_probe_core_membaca_mesin_dari_info(monkeypatch):
    p = probe_core(fake_cmd())
    assert p.problem is None and p.info == CoreInfo("ref_kmeans", False, False, True, 4, 8000000)
    monkeypatch.setenv("FAKE_ENGINES", "omp,cuda")
    cli._PROBE_CACHE.clear()
    assert probe_core(fake_cmd()).info.kmeans_choices == ("auto", "omp", "cuda")


@pytest.mark.parametrize("core", [
    "/tidak/ada/landcover",
    [sys.executable, "-c", "import sys; sys.exit(5)"],
    [sys.executable, "-c", "print('bukan json')"],
    [sys.executable, "-c", "import sys; sys.stderr.write('rusak total'); sys.exit(2)"],
    [],
])
def test_probe_core_tidak_pernah_melempar(core):
    p = probe_core(core)
    assert p.info is None and p.problem


def test_probe_core_berkas_yang_bukan_program(tmp_path):
    f = tmp_path / ("rusak.exe" if os.name == "nt" else "rusak")
    f.write_text("ini bukan program")
    f.chmod(0o755)
    p = probe_core(f)
    assert p.info is None and "tidak bisa dijalankan" in p.problem


def test_probe_core_batas_waktu():
    p = probe_core([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert p.info is None and "tidak selesai" in p.problem


def test_probe_core_cache_per_berkas_dan_kegagalan_tidak_di_cache(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting(cmd, *a, **k):
        calls.append(list(cmd))
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cli.subprocess, "run", counting)
    monkeypatch.setenv("PYTHONPATH", str(config.ROOT / "app"))                # salinan skrip tak punya app/ di dekatnya
    script = tmp_path / "fake.py"
    script.write_text(FAKE.read_text(encoding="utf-8"), encoding="utf-8")
    cmd = [sys.executable, str(script)]
    assert probe_core(cmd).info is not None and probe_core(cmd).info is not None
    assert len(calls) == 1                                                   # panggilan kedua dari cache
    os.utime(script, ns=(1, 1))                                              # berkas berubah: tanya lagi
    assert probe_core(cmd).info is not None and len(calls) == 2
    assert probe_core(tmp_path / "belum_ada").info is None
    assert probe_core(tmp_path / "belum_ada").problem                        # tidak di-cache (tidak ada proses pun)


# --- batas waktu dan fabrik ---------------------------------------------------------- #

def test_resolve_timeout(monkeypatch):
    assert resolve_timeout() == cli.DEFAULT_TIMEOUT_S == 1800.0
    monkeypatch.setenv(cli.TIMEOUT_ENV, "90")
    assert resolve_timeout() == 90.0 and resolve_timeout(5) == 5.0
    monkeypatch.setenv(cli.TIMEOUT_ENV, "   ")
    assert resolve_timeout() == cli.DEFAULT_TIMEOUT_S
    for bad in ("abc", "0", "-5", "inf", "nan"):
        monkeypatch.setenv(cli.TIMEOUT_ENV, bad)
        with pytest.raises(ValueError, match=cli.TIMEOUT_ENV):
            resolve_timeout()
    assert not math.isnan(resolve_timeout(1.5))


def test_make_launcher_cabang_cli(data_dir):
    from launchers import make_launcher
    from launchers.mock import MockLauncher

    lc = make_launcher("cli", data_dir, fake_cmd())
    assert isinstance(lc, CliLauncher) and lc.core == fake_cmd() and lc.data_dir == data_dir
    assert isinstance(make_launcher("mock", data_dir), MockLauncher)
    with pytest.raises(ValueError, match="core_bin"):
        make_launcher("cli", data_dir)
    with pytest.raises(ValueError, match="tidak dikenal"):
        make_launcher("lain", data_dir)


# --- lewat UI (AppTest) ----------------------------------------------------------------- #

@pytest.fixture
def ui(data_dir, runs, monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "cli")
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(config, "CORE_BIN", fake_cmd())
    return runs


def start() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=90).run()
    assert not at.exception, at.exception
    at.sidebar.selectbox(key="f_dataset").set_value("kecil")                # bawaan = folder tile (urut pertama)
    return at


def press(at: AppTest, key: str) -> AppTest:
    at.sidebar.button(key=key).click().run()
    assert not at.exception, at.exception
    return at


def sidebar_captions(at: AppTest) -> list[str]:
    return [c.value for c in at.sidebar.caption]


def test_ui_menyembunyikan_opsi_mesin_yang_tidak_terhubung(ui, monkeypatch):
    at = start()
    assert at.sidebar.selectbox(key="f_kmeans").options == ["Otomatis"]
    assert any("OpenMP dan CUDA belum terhubung" in c for c in sidebar_captions(at))
    assert not at.warning

    monkeypatch.setenv("FAKE_ENGINES", "omp")
    cli._PROBE_CACHE.clear()
    at = start()
    assert at.sidebar.selectbox(key="f_kmeans").options == ["Otomatis", "OpenMP"]
    assert any("CUDA tidak terhubung" in c for c in sidebar_captions(at))

    monkeypatch.setenv("FAKE_ENGINES", "omp,cuda")
    cli._PROBE_CACHE.clear()
    at = start()
    assert at.sidebar.selectbox(key="f_kmeans").options == ["Otomatis", "OpenMP", "CUDA"]
    assert not any("terhubung" in c for c in sidebar_captions(at))


def test_ui_pilihan_mesin_basi_kembali_ke_auto(ui):
    at = AppTest.from_file(APP_FILE, default_timeout=90)
    at.session_state["f_kmeans"] = "cuda"                                    # sisa pilihan dari build lain
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox(key="f_kmeans").value == "auto"


def test_ui_biner_tidak_ada_halaman_tetap_tampil_dengan_peringatan(ui, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CORE_BIN", tmp_path / "tidak_ada" / "landcover")
    at = start()
    assert any("Program landcover belum bisa dipakai" in w.value and "make landcover" in w.value for w in at.warning)
    assert at.sidebar.selectbox(key="f_kmeans").options == ["Otomatis", "OpenMP", "CUDA"]   # tanpa info: semua
    press(at, "btn_run")
    assert len(at.error) == 1 and "make landcover" in at.error[0].value
    assert at.session_state["runs"] == {}


def test_ui_jalankan_cpu_menulis_ke_runs_core_dan_bukan_tiruan(ui):
    at = start()
    press(at, "btn_run")
    assert [e.value for e in at.error] == []
    rid = "kecil__cpu__k8_it20_s42_t1"
    assert list(at.session_state["runs"]) == [rid]
    path = Path(at.session_state["runs"][rid])
    assert path == ui / "core" / rid
    run = read_run(path)
    assert run.mock is False and validate(run) == []
    assert {m.label: m.value for m in at.metric}["Total: CPU saja (L0)"] != "—"
    assert not any("DATA TIRUAN" in m.value for m in at.markdown)
    assert not any("DATA TIRUAN" in w.value for w in at.warning)


def test_ui_bandingkan_dua_run_dan_verifikasi_penuh(ui):
    at = start()
    press(at, "btn_compare")
    assert [e.value for e in at.error] == []
    assert list(at.session_state["runs"]) == ["kecil__cpu__k8_it20_s42_t1", "kecil__opt-auto__k8_it20_s42_t8"]
    cpu, opt = (read_run(p) for p in at.session_state["runs"].values())
    assert cpu.mode == "cpu" and opt.mode == "opt" and cpu.meta["kmeans_device"] == opt.meta["kmeans_device"] == "serial"
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_IDENTICAL and v.source in FULL_SOURCES        # labels.u8 ada: verifikasi penuh
    assert any("identik" in m.value.lower() for m in at.markdown)


@pytest.mark.parametrize("mode, summary, stderr", [
    ("exit1", "gagal saat berjalan", "masukan rusak"),
    ("exit3", "tidak terhubung", "kmeans_cuda tidak terhubung"),
])
def test_ui_kegagalan_biner_menampilkan_ringkasan_dan_stderr(ui, monkeypatch, mode, summary, stderr):
    monkeypatch.setenv("FAKE_CORE", mode)
    at = start()
    press(at, "btn_run")
    assert len(at.error) == 1 and "gagal" in at.error[0].value and summary in at.error[0].value
    assert any(stderr in c.value for c in at.code)                           # stderr di kotak kode
    assert at.session_state["runs"] == {}


def test_ui_keluaran_rusak_menampilkan_daftar_masalah(ui, monkeypatch):
    monkeypatch.setenv("FAKE_CORE", "badjson")
    at = start()
    press(at, "btn_run")
    assert len(at.error) == 1 and "kode 0" in at.error[0].value
    assert any("counts.json" in c.value for c in at.code)
    assert at.session_state["runs"] == {}
