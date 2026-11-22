"""Tes API FastAPI (Fase 1 UI web): backend mock dan cli, tanpa jaringan dan tanpa httpx.

Aplikasi dipanggil langsung lewat klien ASGI kecil (`asgi_client.py`). Backend mock memakai folder
sementara dan MockLauncher berpratinjau kecil supaya cepat; backend cli memakai `fake_core.py`
(skrip Python pengganti program `landcover`, dipakai juga oleh `test_cli_launcher.py`).
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

import asgi_client as http
from api import library
from api.server import create_app
from api.settings import Settings
from conftest import write_run_dir
from landcover.contract import read_run
from launchers.mock import MockLauncher

FAKE = Path(__file__).resolve().parent / "fake_core.py"
DATASET = "tiruan_sleman"          # dataset sintetis (tile 10980x10980 pada model biaya mock)
CPU_LABEL = "CPU saja (L0)"
OPT_LABEL = "Optimalisasi · CUDA"
BODY = {"dataset": DATASET, "k": 4, "iters": 3, "seed": 7}


def mock_factory(**kw):
    def make(backend, data_dir, core_bin):
        return MockLauncher(data_dir, max_side=48, **kw)
    return make


@pytest.fixture
def settings(tmp_path):
    return Settings("mock", tmp_path / "data", tmp_path / "runs", "tidak-dipakai", None)


@pytest.fixture
def app(settings):
    return create_app(settings, launcher_factory=mock_factory())


def run_job(app, path, body=None):
    res = http.post(app, path, json=body or BODY)
    assert res.status == 202, res.text
    data = res.json()
    snap = http.wait_job(app, data["job_id"])
    return data, snap


def compare(app, **body):
    data, snap = run_job(app, "/api/runs/compare", {**BODY, **body})
    assert snap["status"] == "done", snap
    return data


# --- konfigurasi dan dataset ------------------------------------------------------------- #

def test_config_mock(app):
    cfg = http.get(app, "/api/config").json()
    assert cfg["backend"] == "mock" and cfg["mock"] is True
    assert cfg["mock_notice"] == "DATA TIRUAN: bukan hasil pengukuran"
    assert cfg["core"] is None and cfg["active_job"] is None
    assert cfg["defaults"] == {"k": 8, "iters": 20, "seed": 42, "threads": 8, "mode": "cpu", "kmeans": "auto"}
    assert cfg["limits"]["k_max"] == 16 and cfg["limits"]["seed_max"] == 2**32 - 1
    assert cfg["device_colors"] == {"serial": "#8C8C8C", "omp": "#2F6FDE", "cuda": "#4E9A06"}
    assert cfg["labels"]["modes"] == {"cpu": "CPU saja (L0)", "opt": "Optimalisasi"}
    assert cfg["stage_order"] == ["baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"]


def test_datasets_sintetis_muncul_pada_mock_dengan_data_kosong(app):
    data = http.get(app, "/api/datasets").json()
    assert {d["name"] for d in data["datasets"]} == {"tiruan_sleman", "tiruan_kulonprogo"}
    assert all(d["kind"] == "synthetic" for d in data["datasets"])


# --- menjalankan ---------------------------------------------------------------------------- #

def test_run_cpu_lalu_dibaca_dari_disk(app, settings):
    data, snap = run_job(app, "/api/runs", {**BODY, "mode": "cpu"})
    assert snap["status"] == "done" and snap["kind"] == "run"
    assert [s["status"] for s in snap["steps"]] == ["done"]
    [rid] = data["run_ids"]
    assert rid == f"{DATASET}__cpu__k4_it3_s7_t1"
    assert (settings.runs_dir / "mock" / rid / "meta.json").is_file()

    listing = http.get(app, "/api/runs").json()
    assert [r["run_id"] for r in listing["runs"]] == [rid]
    row = listing["runs"][0]
    assert row["mock"] is True and row["mode"] == "cpu" and row["engine_label"] == CPU_LABEL
    assert row["problems"] == [] and row["k"] == 4 and row["total_s"] > 0
    assert listing["unreadable"] == []

    detail = http.get(app, f"/api/runs/{rid}").json()
    assert [s["stage"] for s in detail["stages"]] == ["baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"]
    assert all(s["device"] == "serial" and s["color"] == "#8C8C8C" for s in detail["stages"])
    assert sum(detail["counts"]) == detail["n_pixels"] == detail["width"] * detail["height"]
    assert abs(sum(s["share"] for s in detail["stages"]) - 100.0) < 1e-6
    assert detail["mock_notice"] == "DATA TIRUAN: bukan hasil pengukuran"


def test_bandingkan_menjalankan_cpu_lalu_optimalisasi(app):
    data = compare(app)
    cpu_id, opt_id = data["run_ids"]
    assert cpu_id.endswith("__cpu__k4_it3_s7_t1") and "__opt-auto__" in opt_id

    cmp_ = http.get(app, "/api/compare").json()                  # tanpa petunjuk: kunci run terbaru
    assert cmp_["empty"] is False and cmp_["mock"] is True
    assert cmp_["key"] == {"dataset": DATASET, "k": 4, "iters": 3, "seed": 7}
    cpu, opt = cmp_["cpu"], cmp_["opt"]
    assert cpu["run_id"] == cpu_id and opt["run_id"] == opt_id
    assert cmp_["speedup"] == pytest.approx(cpu["total_s"] / opt["total_s"])
    assert cmp_["speedup"] > 1 and cmp_["speedup_text"] == f"{cmp_['speedup']:.2f}×"

    chart = cmp_["chart"]
    assert chart["labels"] == [CPU_LABEL, OPT_LABEL] and chart["mock"] is True
    assert [d["device"] for d in chart["devices"]] == ["serial", "omp", "cuda"]
    for row, side in zip(chart["rows"], (cpu, opt)):
        assert row["total_s"] == side["total_s"]                 # total resmi timing.json, bukan jumlah tahap
        assert abs(sum(s["share"] for s in row["stages"]) - 100.0) < 1e-6
        assert row["sum_s"] == pytest.approx(sum(s["seconds"] for s in row["stages"]))
    opt_row = chart["rows"][1]
    kmeans = next(s for s in opt_row["stages"] if s["stage"] == "kmeans")
    assert kmeans["device"] == "cuda" and kmeans["color"] == "#4E9A06"
    assert {d["key"] for d in kmeans["detail"]} == {"h2d", "kernel", "d2h"}
    ss = chart["stage_speedup"]
    assert set(ss) == {"baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"}
    cells = {r["label"]: {c["stage"]: c["seconds"] for c in r["stages"]} for r in chart["rows"]}
    assert ss["kmeans"] == pytest.approx(cells[CPU_LABEL]["kmeans"] / cells[OPT_LABEL]["kmeans"])
    assert ss["kmeans"] > 1
    assert chart["stage_speedup_text"]["kmeans"] == f"{ss['kmeans']:.2f}×"
    assert all(r["sum_text"] for r in chart["rows"])
    totals = [r for r in chart["table"] if r["is_total"]]
    assert [t["run"] for t in totals] == [CPU_LABEL, OPT_LABEL] and all(t["percent"] is None for t in totals)
    assert chart["max_seconds"] >= max(r["sum_s"] for r in chart["rows"])


def test_hasil_cpu_tersimpan_dipakai_ulang_kecuali_force(app):
    compare(app)
    again = http.wait_job(app, http.post(app, "/api/runs/compare", json=BODY).json()["job_id"])
    cpu_step, opt_step = again["steps"]
    assert cpu_step["reused"] is True and cpu_step["note"].startswith("hasil tersimpan, ")
    assert opt_step["reused"] is False and opt_step["note"] is None          # Optimalisasi selalu dijalankan
    cpu_id = cpu_step["run_id"]
    assert http.get(app, f"/api/runs/{cpu_id}").json()["note"].startswith("hasil tersimpan")

    forced = http.wait_job(app, http.post(app, "/api/runs/compare", json={**BODY, "force": True}).json()["job_id"])
    assert forced["steps"][0]["reused"] is False
    assert http.get(app, f"/api/runs/{cpu_id}").json()["note"] is None


def test_hanya_satu_pekerjaan_aktif(settings):
    gate = threading.Event()

    class Slow:
        def __init__(self, inner):
            self.inner = inner

        def run(self, params, out_dir, on_progress=None):
            gate.wait(20)
            return self.inner.run(params, out_dir, on_progress=on_progress)

    app = create_app(settings, launcher_factory=lambda b, d, c: Slow(MockLauncher(d, max_side=48)))
    first = http.post(app, "/api/runs", json=BODY)
    assert first.status == 202
    job_id = first.json()["job_id"]
    try:
        busy = http.post(app, "/api/runs/compare", json=BODY)
        assert busy.status == 409 and busy.json()["job_id"] == job_id
        assert "Masih ada run yang berjalan" in busy.json()["detail"]
        assert http.get(app, "/api/config").json()["active_job"] == job_id
        snap = http.get(app, f"/api/jobs/{job_id}").json()
        assert snap["status"] == "running" and snap["steps"][0]["status"] == "running"
    finally:
        gate.set()
    assert http.wait_job(app, job_id)["status"] == "done"
    assert http.get(app, "/api/config").json()["active_job"] is None
    again = http.post(app, "/api/runs", json={**BODY, "seed": 8})                     # bebas lagi
    assert again.status == 202
    http.wait_job(app, again.json()["job_id"])


# --- validasi ------------------------------------------------------------------------------- #

@pytest.mark.parametrize("patch, fragment", [
    ({"k": 1000}, "k harus"),
    ({"k": 0}, "k harus"),
    ({"iters": 0}, "iters harus"),
    ({"seed": -1}, "seed harus"),
    ({"threads": 0}, "threads harus"),
    ({"mode": "gpu"}, "mode harus"),
    ({"kmeans": "tpu"}, "kmeans harus"),
    ({"dataset": "../rahasia"}, "tidak ditemukan"),
    ({"dataset": "tidak_ada"}, "tidak ditemukan"),
    ({"k": "8"}, "k"),
    ({"k": 4.5}, "k"),
    ({"k": True}, "k"),
    ({"abc": 1}, "abc"),
])
def test_isian_ditolak_dengan_422_dan_tanpa_pekerjaan(app, patch, fragment):
    res = http.post(app, "/api/runs", json={**BODY, **patch})
    assert res.status == 422
    assert isinstance(res.json()["detail"], str) and fragment in res.json()["detail"]
    assert http.get(app, "/api/config").json()["active_job"] is None
    assert http.get(app, "/api/runs").json()["runs"] == []


def test_badan_tanpa_dataset_ditolak(app):
    res = http.post(app, "/api/runs", json={"k": 4})
    assert res.status == 422 and "dataset" in res.json()["detail"]


# --- galat saat run ------------------------------------------------------------------------- #

def test_galat_launcher_jadi_pekerjaan_error_dan_langkah_berikutnya_dilewati(settings):
    class FailsOnOpt:
        def __init__(self, inner):
            self.inner = inner

        def run(self, params, out_dir, on_progress=None):
            if params.mode == "opt":
                raise RuntimeError("GPU hilang")
            return self.inner.run(params, out_dir, on_progress=on_progress)

    app = create_app(settings, launcher_factory=lambda b, d, c: FailsOnOpt(MockLauncher(d, max_side=48)))
    _, snap = run_job(app, "/api/runs/compare")
    assert snap["status"] == "error"
    assert [s["status"] for s in snap["steps"]] == ["done", "error"]
    assert snap["error"]["message"] == "Run 'Optimalisasi' gagal: RuntimeError: GPU hilang"
    assert snap["error"]["step"] == 1
    assert len(http.get(app, "/api/runs").json()["runs"]) == 1       # run CPU tetap tersimpan dan tampil
    cmp_ = http.get(app, "/api/compare").json()
    assert cmp_["cpu"] is not None and cmp_["opt"] is None and cmp_["speedup"] is None
    assert cmp_["chart"]["stage_speedup"] is None and len(cmp_["chart"]["rows"]) == 1
    assert cmp_["speedup_text"] == "—"

    _, snap = run_job(app, "/api/runs", {**BODY, "mode": "cpu", "seed": 99})
    assert snap["status"] == "done"                                    # pekerjaan gagal tidak mengunci server

    assert http.get(app, f"/api/jobs/{snap['id']}/events").events()[-1][0] == "done"
    failed_id = http.post(app, "/api/runs/compare", json={**BODY, "seed": 100}).json()["job_id"]
    failed = http.wait_job(app, failed_id)
    assert failed["status"] == "error"
    names = [n for n, _ in http.get(app, f"/api/jobs/{failed_id}/events").events()]
    assert names[-1] == "failed" and "error" not in names and "done" not in names


def test_launcher_tidak_bisa_dibuat_jadi_422(settings):
    def broken(backend, data_dir, core_bin):
        raise ValueError("backend cli butuh core_bin")

    app = create_app(settings, launcher_factory=broken)
    res = http.post(app, "/api/runs", json=BODY)
    assert res.status == 422 and "core_bin" in res.json()["detail"]
    assert http.get(app, "/api/config").json()["active_job"] is None


# --- SSE ------------------------------------------------------------------------------------ #

def test_sse_mengalirkan_peristiwa_dan_selalu_ditutup_dengan_done(app):
    data = http.post(app, "/api/runs/compare", json=BODY).json()
    res = http.get(app, f"/api/jobs/{data['job_id']}/events")        # dibaca sampai server menutup respons
    assert res.status == 200 and res.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in res.events()]
    assert names[0] == "snapshot" and names[-1] == "done" and names.count("done") == 1
    steps = [(d["index"], d["status"]) for n, d in res.events() if n == "step"]
    assert steps == [(0, "running"), (0, "done"), (1, "running"), (1, "done")]
    final = res.events()[-1][1]["job"]
    assert final["status"] == "done" and final["run_ids"] == data["run_ids"]
    assert "retry: " in res.text


def test_sse_pekerjaan_berjalan_mengirim_tick_dan_baris_progres(settings):
    gate = threading.Event()

    class Gated:
        def __init__(self, inner):
            self.inner = inner

        def run(self, params, out_dir, on_progress=None):
            if on_progress:
                on_progress("[stage] baca mulai")
            gate.wait(20)
            return self.inner.run(params, out_dir, on_progress=on_progress)

    app = create_app(settings, launcher_factory=lambda b, d, c: Gated(MockLauncher(d, max_side=48)))
    job_id = http.post(app, "/api/runs", json=BODY).json()["job_id"]
    box: dict = {}
    reader = threading.Thread(target=lambda: box.update(res=http.get(app, f"/api/jobs/{job_id}/events")))
    reader.start()
    time.sleep(0.7)                                  # cukup untuk beberapa tick
    gate.set()
    reader.join(30)
    assert not reader.is_alive()
    events = box["res"].events()
    kinds = [n for n, _ in events]
    assert "tick" in kinds and kinds[-1] == "done"
    ticks = [d for n, d in events if n == "tick"]
    assert all(t["elapsed_s"] >= 0 for t in ticks) and ticks[-1]["step"] == 0
    assert any(d["line"] == "[stage] baca mulai" for n, d in events if n == "progress")


def test_job_tidak_dikenal_404(app):
    assert http.get(app, "/api/jobs/tidak-ada").status == 404
    assert http.get(app, "/api/jobs/tidak-ada/events").status == 404


# --- pustaka hasil -------------------------------------------------------------------------- #

def test_folder_rusak_dilaporkan_dan_tidak_menggagalkan_daftar(app, settings):
    compare(app)
    bad = settings.runs_dir / "mock" / "rusak__cpu__k4_it3_s7_t1"
    bad.mkdir(parents=True)
    (bad / "meta.json").write_text("{bukan json", encoding="utf-8")
    (settings.runs_dir / "mock" / "bukan-run").mkdir()                       # tanpa meta.json: dilewati diam-diam
    listing = http.get(app, "/api/runs").json()
    assert len(listing["runs"]) == 2
    assert [u["run_id"] for u in listing["unreadable"]] == ["rusak__cpu__k4_it3_s7_t1"]
    res = http.get(app, "/api/runs/rusak__cpu__k4_it3_s7_t1")
    assert res.status == 404 and "tidak terbaca" in res.json()["detail"]


def test_run_id_tidak_pernah_menjadi_path(app, settings, tmp_path):
    compare(app)
    (tmp_path / "rahasia.json").write_text("{}", encoding="utf-8")
    for rid in ("..", "../runs/mock", "%2E%2E%2Frahasia", "..%5Crahasia", "C:/Windows"):
        assert http.get(app, f"/api/runs/{rid}").status in (404, 405)
    assert http.get(app, "/api/verify", cpu="../x", opt="../y").status == 404
    assert http.get(app, "/api/compare", cpu="../x").status == 422


def test_hasil_bertahan_antar_server(settings):
    first = create_app(settings, launcher_factory=mock_factory())
    compare(first)
    second = create_app(settings, launcher_factory=mock_factory())            # server baru, folder sama
    assert len(http.get(second, "/api/runs").json()["runs"]) == 2
    assert http.get(second, "/api/compare").json()["speedup"] > 1


def test_kelompok_dan_pemilihan_pasangan(app):
    compare(app)
    compare(app, seed=8)
    listing = http.get(app, "/api/runs").json()
    assert len(listing["groups"]) == 2 and all(g["complete"] for g in listing["groups"])
    assert [g["seed"] for g in listing["groups"]] == [8, 7]                  # kelompok terbaru di atas
    default = http.get(app, "/api/compare").json()
    assert default["key"]["seed"] == 8
    chosen = http.get(app, "/api/compare", dataset=DATASET, k=4, iters=3, seed=7).json()
    assert chosen["key"]["seed"] == 7 and chosen["cpu"]["run_id"].endswith("_s7_t1")
    assert chosen["opt"]["run_id"].endswith("_s7_t8")


def test_pasangan_hanya_sebanding(app):
    other = compare(app, seed=8)["run_ids"]
    compare(app)
    cpu_seed7 = f"{DATASET}__cpu__k4_it3_s7_t1"
    assert http.get(app, "/api/compare", cpu=cpu_seed7, opt=other[1]).status == 422
    assert http.get(app, "/api/compare", cpu=other[1]).status == 422          # mode salah
    assert http.get(app, "/api/compare", dataset=DATASET, k=4).status == 422  # kunci tidak lengkap
    assert http.get(app, "/api/compare", dataset=DATASET, k=5, iters=3, seed=7).json()["empty"] is True
    assert http.get(app, "/api/compare", dataset=DATASET, k=4, iters=3, seed=7, opt=other[1]).status == 422


def test_perpustakaan_kosong(app):
    cmp_ = http.get(app, "/api/compare").json()
    assert cmp_["empty"] is True and cmp_["cpu"] is None and cmp_["opt"] is None and cmp_["chart"] is None
    assert http.get(app, "/api/runs").json() == {"runs": [], "unreadable": [], "groups": []}


def test_dua_run_optimalisasi_dipilih_eksplisit(app):
    compare(app)                                                              # opt-auto (cuda pada tile penuh)
    omp = {**BODY, "mode": "opt", "kmeans": "omp", "threads": 4}
    _, snap = run_job(app, "/api/runs", omp)
    assert snap["status"] == "done"
    latest = http.get(app, "/api/compare").json()
    assert latest["opt"]["run_id"].endswith("__opt-omp__k4_it3_s7_t4")        # run Optimalisasi TERBARU
    auto_id = f"{DATASET}__opt-auto__k4_it3_s7_t8"
    chosen = http.get(app, "/api/compare", opt=auto_id).json()
    assert chosen["opt"]["run_id"] == auto_id
    assert chosen["chart"]["labels"] == [CPU_LABEL, OPT_LABEL]


def test_nilai_tak_hingga_di_timing_menjadi_null_bukan_galat_server(app, settings):
    rid = f"{DATASET}__cpu__k4_it3_s7_t1"
    write_run_dir(
        settings.runs_dir / "mock" / rid, width=16, height=12, k=4,
        meta={"dataset": DATASET, "mode": "cpu", "kmeans_device": "serial", "iters": 3, "seed": 7},
        timing={"total_ms": float("nan"), "kmeans_ms": float("inf")},
    )
    cmp_ = http.get(app, "/api/compare")
    assert cmp_.status == 200
    row = cmp_.json()["chart"]["rows"][0]
    assert row["total_s"] is None and row["total_text"] is None
    assert all(math.isfinite(s["seconds"]) for s in row["stages"])
    assert http.get(app, f"/api/runs/{rid}").status == 200
    json.loads(cmp_.text)                                                     # JSON sah (tanpa NaN)


# --- verifikasi ----------------------------------------------------------------------------- #

def test_verifikasi_pratinjau_adalah_parsial_bukan_seratus_persen_resolusi_penuh(app):
    cpu_id, opt_id = compare(app)["run_ids"]
    v = http.get(app, "/api/verify", cpu=cpu_id, opt=opt_id).json()
    assert v["status"] == "identical" and v["source"] == "preview" and v["full"] is False
    assert v["level"] == "partial" and v["icon"] == "ℹ️"
    assert v["verdict"].startswith("Verifikasi parsial: label pratinjau (")
    assert "Label identik 100 %" not in v["verdict"]
    assert v["mock"] is True and "tidak membuktikan apa pun tentang core" in v["mock_notice"]
    assert v["pair_label"] == f"{CPU_LABEL} ↔ {OPT_LABEL}"
    assert [r["what"] for r in v["rows"]] == ["Label pratinjau", "Centroid", "Counts", "Sumber"]


def test_verifikasi_berbeda_pada_pratinjau_yang_diubah(settings):
    app = create_app(settings, launcher_factory=mock_factory(break_verification=True))
    cpu_id, opt_id = compare(app)["run_ids"]
    v = http.get(app, "/api/verify", cpu=cpu_id, opt=opt_id).json()
    assert v["status"] == "different" and v["level"] == "bad" and v["icon"] == "❌"
    assert v["verdict"].startswith("Berbeda: ") and "label pratinjau identik" in v["verdict"]
    assert v["identical_pct"] < 100 and v["identical_pct_text"] != "100"


def test_verifikasi_resolusi_penuh_dipotong_ke_bawah_dan_hijau_hanya_bila_identik(app, settings):
    settings.data_dir.mkdir(parents=True)
    Image.effect_noise((40, 30), 80).convert("RGB").save(settings.data_dir / "kecil.png")
    cpu_id, opt_id = compare(app, dataset="kecil")["run_ids"]
    cpu_dir, opt_dir = (settings.runs_dir / "mock" / r for r in (cpu_id, opt_id))
    run = read_run(cpu_dir)
    assert (run.width, run.height) == run.labels_preview.shape[::-1]         # pratinjau = resolusi penuh (citra kecil)
    full = bytes(run.labels_preview.astype("uint8").ravel())
    (cpu_dir / "labels.u8").write_bytes(full)
    (opt_dir / "labels.u8").write_bytes(full)
    v = http.get(app, "/api/verify", cpu=cpu_id, opt=opt_id).json()
    assert v["full"] is True and v["level"] == "ok" and v["icon"] == "✅"
    assert v["verdict"] == "Label identik 100 % (1,200 piksel), centroid sama"
    broken = bytearray(full)
    broken[-1] = (broken[-1] + 1) % 4
    (opt_dir / "labels.u8").write_bytes(bytes(broken))
    v = http.get(app, "/api/verify", cpu=cpu_id, opt=opt_id).json()          # kunci cache berubah: dibaca ulang
    assert v["status"] == "different" and v["n_diff"] == 1
    assert v["identical_pct"] < 100 and v["identical_pct_text"] != "100"
    assert v["verdict"].startswith("Berbeda: 99.91 % label identik (1 dari 1,200 piksel berbeda)")


def test_verifikasi_dua_run_yang_tidak_sebanding_diberi_alasan(app):
    a = compare(app)["run_ids"]
    b = compare(app, k=5)["run_ids"]
    v = http.get(app, "/api/verify", cpu=a[0], opt=b[1]).json()
    assert v["status"] == "not_comparable" and v["level"] == "warn"
    assert v["verdict"].startswith("Tidak sebanding: ") and "K berbeda" in v["verdict"]
    assert v["rows"] == [] and v["source"] is None


# --- frontend hasil build ---------------------------------------------------------------------- #

def test_menyajikan_frontend_dan_tidak_menelan_404_api(settings, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Kinerja</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "rahasia.txt").write_text("JANGAN BOCOR", encoding="utf-8")
    app = create_app(Settings("mock", settings.data_dir, settings.runs_dir, "x", dist), launcher_factory=mock_factory())

    assert "<title>Kinerja</title>" in http.get(app, "/").text
    assert "<title>Kinerja</title>" in http.get(app, "/kinerja").text           # rute sisi klien
    assert http.get(app, "/assets/app.js").text == "console.log(1)"
    assert http.get(app, "/favicon.svg").text == "<svg/>"
    res = http.get(app, "/api/tidak-ada")
    assert res.status == 404 and res.json()["detail"] == "rute API tidak ada"   # bukan index.html
    assert http.get(app, "/api/config").status == 200
    for sneaky in ("/../rahasia.txt", "/..%2Frahasia.txt", "/%2e%2e/rahasia.txt", "/assets/../../rahasia.txt"):
        assert "JANGAN BOCOR" not in http.get(app, sneaky).text


def test_tanpa_build_root_memberi_petunjuk(app):
    res = http.get(app, "/")
    assert res.status == 404 and "npm run build" in res.json()["detail"]


# --- backend cli (fake_core) ------------------------------------------------------------------- #

@pytest.fixture
def cli_app(tmp_path, monkeypatch):
    from launchers import cli

    for name in ("FAKE_CORE", "FAKE_CORE_DROP", "FAKE_ENGINES", "FAKE_CORE_LOG"):
        monkeypatch.delenv(name, raising=False)
    cli._PROBE_CACHE.clear()                         # cache --info berkunci berkas skrip, bukan env FAKE_*
    data = tmp_path / "data"
    data.mkdir()
    Image.effect_noise((40, 30), 80).convert("RGB").save(data / "sleman.png")
    settings = Settings("cli", data, tmp_path / "runs", [sys.executable, str(FAKE)], None)
    yield create_app(settings), settings
    cli._PROBE_CACHE.clear()


CLI_BODY = {"dataset": "sleman", "k": 3, "iters": 2, "seed": 5, "threads": 2}


def test_cli_config_melaporkan_mesin_yang_terhubung(cli_app, monkeypatch):
    app, _ = cli_app
    cfg = http.get(app, "/api/config").json()
    assert cfg["backend"] == "cli" and cfg["mock"] is False and cfg["mock_notice"] is None
    assert cfg["core"]["ok"] is True and cfg["core"]["kmeans_choices"] == ["auto"]
    assert cfg["core"]["info"]["cpu"] == "ref_kmeans" and cfg["core"]["info"]["omp"] is False
    assert "belum terhubung" in cfg["core"]["notice"]
    assert [d["name"] for d in http.get(app, "/api/datasets").json()["datasets"]] == ["sleman"]


def test_cli_biner_hilang_dilaporkan_bukan_dibiarkan_mogok(tmp_path, monkeypatch):
    from launchers import cli

    cli._PROBE_CACHE.clear()
    settings = Settings("cli", tmp_path / "data", tmp_path / "runs", tmp_path / "tidak-ada.exe", None)
    app = create_app(settings)
    cfg = http.get(app, "/api/config").json()
    assert cfg["core"]["ok"] is False and cfg["core"]["problem"]
    assert cfg["core"]["kmeans_choices"] == ["auto", "omp", "cuda"]


def test_cli_kmeans_yang_tidak_terhubung_ditolak_sebelum_run(cli_app):
    app, _ = cli_app
    res = http.post(app, "/api/runs", json={**CLI_BODY, "mode": "opt", "kmeans": "cuda"})
    assert res.status == 422 and "tidak terhubung" in res.json()["detail"] and "tersedia: auto" in res.json()["detail"]
    assert http.post(app, "/api/runs/compare", json={**CLI_BODY, "kmeans": "omp"}).status == 422
    ok = http.post(app, "/api/runs", json={**CLI_BODY, "mode": "cpu", "kmeans": "cuda"})   # cpu mengabaikannya
    assert ok.status == 202
    assert http.wait_job(app, ok.json()["job_id"])["status"] == "done"


def test_cli_bandingkan_dengan_program_pengganti_lolos_verifikasi_penuh(cli_app):
    app, settings = cli_app
    data, snap = run_job(app, "/api/runs/compare", CLI_BODY)
    assert snap["status"] == "done", snap
    cpu_id, opt_id = data["run_ids"]
    assert (settings.runs_dir / "core" / cpu_id / "meta.json").is_file()      # backend cli -> runs/core/
    cmp_ = http.get(app, "/api/compare").json()
    assert cmp_["mock"] is False and cmp_["mock_notice"] is None
    assert cmp_["chart"]["mock"] is False and cmp_["cpu"]["mock"] is False
    assert cmp_["cpu"]["problems"] == [] and cmp_["opt"]["problems"] == []
    assert cmp_["cpu"]["has_labels"] and cmp_["opt"]["has_labels"]            # CliLauncher selalu --write-labels
    v = http.get(app, "/api/verify", cpu=cpu_id, opt=opt_id).json()
    assert v["status"] == "identical" and v["full"] is True and v["level"] == "ok"
    assert v["verdict"].startswith("Label identik 100 % (1,200 piksel)")
    assert v["mock"] is False and v["mock_notice"] is None


def test_cli_galat_program_tampil_dengan_ringkasan_dan_detail(cli_app, monkeypatch):
    app, _ = cli_app
    monkeypatch.setenv("FAKE_CORE", "exit1")
    _, snap = run_job(app, "/api/runs", {**CLI_BODY, "mode": "cpu"})
    assert snap["status"] == "error"
    assert snap["error"]["message"].startswith("Run 'CPU saja (L0)' gagal: ")
    assert snap["error"]["detail"]                                              # ekor stderr program
    assert http.get(app, "/api/runs").json()["runs"] == []


# --- library.select_pair langsung ---------------------------------------------------------------- #

def test_select_pair_perpustakaan_kosong_dan_run_tidak_ada(tmp_path):
    lib = library.scan_library(tmp_path, "mock")
    assert library.select_pair(lib) == (None, None)
    assert library.scan_library(tmp_path / "tidak-ada", "cli").runs == {}
