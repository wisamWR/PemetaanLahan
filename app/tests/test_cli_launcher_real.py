"""Tes CliLauncher dengan program `landcover` NYATA (hasil `make landcover`), pada citra 2048x2048.

Dilewati bila biner tidak ada atau tidak bisa dijalankan di sini. Di WSL/Linux biner dibaca dari
LC_CORE_BIN (bawaan build/landcover). Dari Windows, arahkan LC_TEST_CORE_BIN ke pembungkus yang
meneruskan ke WSL (terjemahan path C:\\... ke /mnt/c/...; contoh ada di catatan PROGRESS Langkah 11):

    set LC_TEST_CORE_BIN=...\\lc_wsl_stream.cmd
    .venv\\Scripts\\python.exe -m pytest app/tests/test_cli_launcher_real.py -q

Yang diuji di sini adalah PENGHUBUNGNYA (launcher, folder run, UI). Kebenaran K-Means program itu
sendiri (eksak terhadap reference_kmeans.py) diuji `test_landcover_cli.py`. Mode omp dan cuda hanya
diuji bila `--info` melaporkannya terhubung.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from landcover.contract import read_run, validate
from landcover.verify import FULL_SOURCES, STATUS_IDENTICAL, verify_pair
from launchers.base import RunParams, out_dir_for
from launchers.cli import CliLauncher, CoreFailedError, probe_core

SIDE = 2048
APP_FILE = str(config.ROOT / "app" / "streamlit_app.py")
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
        pytest.skip("biner landcover tidak ada/tidak bisa dijalankan di sini (make landcover; lihat LC_TEST_CORE_BIN)")
    return p


@pytest.fixture(scope="module")
def info(core):
    probe = probe_core(core)
    assert probe.problem is None, probe.problem
    return probe.info


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    """Folder data/ sementara: citra 2048x2048 terstruktur, dan satu PNG terpotong (rusak)."""
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(11)
    palette = rng.integers(0, 256, (7, 3))
    idx = rng.integers(0, 7, (SIDE // 16, SIDE // 16))
    blocks = np.kron(idx, np.ones((16, 16), dtype=int))
    img = np.clip(palette[blocks] + rng.integers(-12, 13, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(d / "foto2048.png", compress_level=1)
    (d / "rusak.png").write_bytes((d / "foto2048.png").read_bytes()[:300])
    return d


@pytest.fixture
def runs(tmp_path) -> Path:
    return tmp_path / "runs"


def launcher(data, core) -> CliLauncher:
    return CliLauncher(data, core, timeout=900)


def P(**kw) -> RunParams:
    base = dict(dataset="foto2048", mode="cpu", kmeans="auto", k=5, iters=3, seed=7, threads=8)
    base.update(kw)
    return RunParams(**base)


def test_info_nyata_konsisten_dengan_pilihan_ui(info):
    assert info.cpu in ("kmeans_cpu", "ref_kmeans")
    assert info.kmeans_choices[0] == "auto"
    assert ("omp" in info.kmeans_choices) == info.omp and ("cuda" in info.kmeans_choices) == info.cuda
    assert (info.notice is None) == (info.omp and info.cuda)


def test_cpu_dan_opt_2048_valid_waktu_dari_core_dan_label_identik(core, data, runs, info):
    results = {}
    for params in (P(mode="cpu"), P(mode="opt")):
        out = out_dir_for(runs, "cli", params)
        lines: list[str] = []
        assert launcher(data, core).run(params, out, lines.append) == out
        stages = [ln for ln in lines if ln.startswith("[stage]")]
        assert [s.split()[1] for s in stages] == STAGES
        assert any(ln.startswith("[info] mesin K-Means") for ln in lines)
        run = read_run(out)
        assert validate(run) == []
        m = run.meta
        assert m["mock"] is False and m["dataset"] == "foto2048" and (m["width"], m["height"]) == (SIDE, SIDE)
        assert run.total > 0 and all(run.stages[s].time >= 0 for s in STAGES) and m["host"]
        assert int(run.counts.sum()) == SIDE * SIDE
        assert (out / "labels.u8").stat().st_size == SIDE * SIDE         # --write-labels selalu
        results[params.mode] = run

    cpu, opt = results["cpu"], results["opt"]
    assert cpu.meta["kmeans_device"] == "serial"
    # `auto`: cuda bila N*K >= ambang kerja dari --info (#12; biner lama: N >= ambang piksel) dan cuda
    # terhubung, selain itu omp, lalu serial
    n, k = SIDE * SIDE, P(mode="opt").k
    if info.auto_cuda_threshold_work is not None:
        want_cuda = n * k >= info.auto_cuda_threshold_work
    else:
        want_cuda = info.auto_cuda_threshold_pixels is not None and n >= info.auto_cuda_threshold_pixels
    if want_cuda and info.cuda:
        expected = "cuda"
    else:
        expected = "omp" if info.omp else "serial"
    assert opt.meta["kmeans_device"] == expected
    assert "kmeans_auto_reason" in opt.meta
    assert np.array_equal(cpu.centroids, opt.centroids) and np.array_equal(cpu.counts, opt.counts)
    assert (cpu.path / "labels.u8").read_bytes() == (opt.path / "labels.u8").read_bytes()
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_IDENTICAL and v.source in FULL_SOURCES


def test_mesin_eksplisit_yang_tidak_terhubung_kode_3_dengan_stderr_asli(core, data, runs, info):
    engine = next((e for e in ("cuda", "omp") if not getattr(info, e)), None)
    if engine is None:
        pytest.skip("omp dan cuda terhubung pada build ini")
    params = P(mode="opt", kmeans=engine)
    with pytest.raises(CoreFailedError) as ei:
        launcher(data, core).run(params, out_dir_for(runs, "cli", params))
    e = ei.value
    assert e.exit_code == 3 and "tidak terhubung" in e.summary and "tidak terhubung pada build ini" in e.detail
    assert not (out_dir_for(runs, "cli", params) / "meta.json").exists()


def test_cuda_eksplisit_perangkat_cuda_dan_identik_dengan_cpu(core, data, runs, info):
    """cuda dipilih eksplisit (tidak lewat `auto`), supaya uji ini tidak bergantung pada ambang."""
    if not info.cuda:
        pytest.skip("cuda tidak terhubung pada build ini")
    by_mode = {}
    for params in (P(mode="cpu"), P(mode="opt", kmeans="cuda")):
        out = out_dir_for(runs, "cli", params)
        assert launcher(data, core).run(params, out) == out
        by_mode[params.mode] = read_run(out)
    cpu, opt = by_mode["cpu"], by_mode["opt"]
    assert validate(opt) == []
    assert opt.meta["kmeans_device"] == "cuda" and opt.meta["kmeans_impl"] == "kmeans_cuda"
    timing = json.loads((opt.path / "timing.json").read_text(encoding="utf-8"))
    assert timing["h2d_ms"] > 0 and timing["kernel_ms"] > 0 and timing["d2h_ms"] > 0
    assert np.array_equal(cpu.centroids, opt.centroids) and np.array_equal(cpu.counts, opt.counts)
    assert (cpu.path / "labels.u8").read_bytes() == (opt.path / "labels.u8").read_bytes()
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_IDENTICAL and v.source in FULL_SOURCES


def test_png_rusak_kode_1_dan_folder_tidak_valid(core, data, runs):
    params = P(dataset="rusak")
    out = out_dir_for(runs, "cli", params)
    with pytest.raises(CoreFailedError) as ei:
        launcher(data, core).run(params, out)
    assert ei.value.exit_code == 1 and ei.value.detail                      # stderr asli ditampilkan
    assert not (out / "meta.json").exists()


@pytest.fixture
def ui(data, runs, core, monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "cli")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(config, "CORE_BIN", core)
    return runs


def test_ui_bandingkan_2048_dua_run_nyata_dan_verifikasi_penuh(ui, info):
    at = AppTest.from_file(APP_FILE, default_timeout=900).run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox(key="f_kmeans").options[0] == "Otomatis"
    assert len(at.sidebar.selectbox(key="f_kmeans").options) == len(info.kmeans_choices)
    at.sidebar.selectbox(key="f_dataset").set_value("foto2048")
    at.sidebar.button(key="btn_compare").click().run()
    assert not at.exception, at.exception
    assert [e.value for e in at.error] == []
    assert list(at.session_state["runs"]) == ["foto2048__cpu__k8_it20_s42_t1", "foto2048__opt-auto__k8_it20_s42_t8"]
    cpu, opt = (read_run(p) for p in at.session_state["runs"].values())
    assert cpu.mock is False and opt.mock is False and validate(cpu) == [] and validate(opt) == []
    assert str(cpu.path).startswith(str(ui / "core"))                       # runs/core/, bukan runs/mock/
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_IDENTICAL and v.source in FULL_SOURCES
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total: CPU saja (L0)"] != "—" and metrics["Total: Optimalisasi"] != "—"
    assert not any("DATA TIRUAN" in m.value for m in at.markdown)
    assert any("identik" in m.value.lower() for m in at.markdown)           # lencana verifikasi muncul
