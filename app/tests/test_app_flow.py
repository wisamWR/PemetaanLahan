"""Alur layar utama dengan AppTest (tanpa server dan browser): plan Langkah 6, "Selesai bila".

Semua tes memakai folder data/ dan runs/ sementara (config ditimpa), jadi repo tidak tersentuh.
Dataset "kecil" = satu PNG 40x30, sehingga tiap run hanya beberapa puluh milidetik.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from ui.compare import MOCK_NOTICE

APP_FILE = str(config.ROOT / "app" / "streamlit_app.py")
TIMEOUT = 60                     # bawaan AppTest 3 detik terlalu pendek untuk tes yang menjalankan run

CPU_ID = "kecil__cpu__k8_it20_s42_t1"
OPT_ID = "kecil__opt-auto__k8_it20_s42_t8"
K5_CPU_ID = "kecil__cpu__k5_it20_s42_t1"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(data / "kecil.png")
    monkeypatch.setattr(config, "BACKEND", "mock")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.delenv("LC_MOCK_SLEEP", raising=False)
    return data, runs


def start() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()
    assert not at.exception
    return at


def press(at: AppTest, key: str) -> AppTest:
    at.sidebar.button(key=key).click().run()
    assert not at.exception, at.exception
    return at


def run_mode(at: AppTest, mode: str) -> AppTest:
    at.sidebar.radio(key="f_mode").set_value(mode)
    return press(at, "btn_run")


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def no_errors(at: AppTest) -> None:
    assert [e.value for e in at.error] == []


# --- halaman awal ------------------------------------------------------------ #

def test_halaman_awal_tanpa_hasil(sandbox):
    at = start()
    assert [t.value for t in at.title] == ["Pemetaan Tutupan Lahan"]
    assert at.sidebar.selectbox(key="f_dataset").options[0] == "kecil (PNG)"
    assert any("sintetis" in o for o in at.sidebar.selectbox(key="f_dataset").options)
    assert any("Belum ada hasil" in i.value for i in at.info)
    assert not at.metric and not at.get("image")
    assert at.session_state["runs"] == {} and at.session_state["class_maps"] == {}
    assert at.sidebar.button(key="upl_save").disabled                   # tombol simpan mati tanpa berkas
    assert at.sidebar.slider(key="f_k").value == 8
    assert at.sidebar.number_input(key="f_iters").value == 20
    assert at.sidebar.number_input(key="f_seed").value == 42
    assert at.sidebar.slider(key="f_threads").value == 8
    assert at.sidebar.radio(key="f_mode").value == "cpu"
    assert at.sidebar.selectbox(key="f_kmeans").value == "auto"


def test_run_ditulis_ke_folder_sandbox_bukan_ke_repo(sandbox):
    data, runs = sandbox
    in_repo = config.ROOT / "runs" / "mock" / CPU_ID
    existed = in_repo.exists()
    at = start()
    run_mode(at, "cpu")
    assert (runs / "mock" / CPU_ID / "meta.json").is_file()
    assert in_repo.exists() == existed                                      # repo tidak tersentuh


# --- alur "Selesai bila" ------------------------------------------------------ #

def test_alur_lengkap_dan_hasil_tidak_hilang_saat_widget_berubah(sandbox):
    data, runs = sandbox
    at = start()
    assert at.sidebar.selectbox(key="f_dataset").value == "kecil"

    # 1. Jalankan CPU saja
    run_mode(at, "cpu")
    no_errors(at)
    assert list(at.session_state["runs"]) == [CPU_ID]
    assert at.session_state["active_run"] == CPU_ID
    m = metrics(at)
    assert m["Total: CPU saja (L0)"] != "—" and m["Total: Optimalisasi"] == "—" and m["Speedup (T_cpu / T_opt)"] == "—"
    assert MOCK_NOTICE in [w.value for w in at.warning]                  # lencana di atas halaman
    assert any("DATA TIRUAN" in s.value for s in at.subheader)           # lencana di judul peta
    assert MOCK_NOTICE in captions(at)                                   # kalimat penuh di bawah peta
    assert any("DATA TIRUAN" in c for c in captions(at) if c != MOCK_NOTICE)    # dan di bawah kartu
    assert len(at.get("image")) == 2                                      # citra asli + peta kelas
    assert not any("belum dibuat" in c for c in captions(at))            # semua slot (Langkah 7-9) sudah terisi
    assert any(s.value.startswith("Status verifikasi antarmode") for s in at.subheader)
    assert (runs / "mock" / CPU_ID / "meta.json").is_file()

    # 2. Jalankan Optimalisasi: kartu lengkap dengan speedup
    run_mode(at, "opt")
    no_errors(at)
    assert list(at.session_state["runs"]) == [CPU_ID, OPT_ID]
    assert at.session_state["active_run"] == OPT_ID
    m = metrics(at)
    assert re.fullmatch(r"\d+\.\d\d s", m["Total: Optimalisasi"]) and re.fullmatch(r"\d+\.\d\d×", m["Speedup (T_cpu / T_opt)"])
    cpu_run, opt_run = (at.session_state["runs"][r] for r in (CPU_ID, OPT_ID))
    assert cpu_run != opt_run

    # 3. Bandingkan: CPU dipakai ulang dari runs/, Optimalisasi dijalankan lagi, aktif = opt
    before = (runs / "mock" / CPU_ID / "meta.json").stat().st_mtime_ns
    press(at, "btn_compare")
    no_errors(at)
    assert at.session_state["active_run"] == OPT_ID
    assert list(at.session_state["runs"]) == [CPU_ID, OPT_ID]            # CPU didaftarkan dulu, opt terakhir
    assert any(c.startswith("hasil tersimpan, ") for c in captions(at))
    assert (runs / "mock" / CPU_ID / "meta.json").stat().st_mtime_ns == before   # tidak ditulis ulang
    assert "—" not in metrics(at).values()
    shown = metrics(at)

    # 4. Ganti K tanpa menekan tombol: tidak ada run baru dan hasil tetap tampil
    at.sidebar.slider(key="f_k").set_value(5)
    at.run()
    assert not at.exception
    assert list(at.session_state["runs"]) == [CPU_ID, OPT_ID]
    assert metrics(at) == shown and len(at.get("image")) == 2

    # 5. Ganti warna peta: hasil tetap tampil
    at.radio(key="map_view").set_value("class")
    at.run()
    assert not at.exception and metrics(at) == shown and len(at.get("image")) == 2

    # 6. Kirim dengan K baru: run baru muncul, run lama tetap bisa dipilih.
    #    (AppTest hanya meneruskan nilai form yang diset tepat sebelum tombol ditekan.)
    at.sidebar.slider(key="f_k").set_value(5)
    run_mode(at, "cpu")
    no_errors(at)
    assert list(at.session_state["runs"]) == [CPU_ID, OPT_ID, K5_CPU_ID]
    assert at.session_state["active_run"] == K5_CPU_ID
    m = metrics(at)
    assert m["Total: CPU saja (L0)"] != "—" and m["Total: Optimalisasi"] == "—"     # K=5 belum punya pasangan opt
    pick = at.selectbox(key="active_run_pick")
    assert len(pick.options) == 3 and "K=5" in pick.options[0]              # terbaru di atas
    assert pick.value == K5_CPU_ID                                           # pemilih menunjuk run baru
    pick.select_index(1).run()                                               # pilih run lama (OPT K=8)
    assert not at.exception
    assert at.session_state["active_run"] == OPT_ID
    assert at.selectbox(key="active_run_pick").value == OPT_ID
    assert metrics(at) == shown                                              # kartu mengikuti run terpilih

    # 7. Jalankan CPU saja (K=5) lagi: "hasil tersimpan"; dengan "paksa jalankan ulang": hitung ulang
    meta = runs / "mock" / K5_CPU_ID / "meta.json"
    stamp = meta.stat().st_mtime_ns
    at.sidebar.slider(key="f_k").set_value(5)
    run_mode(at, "cpu")
    assert any(c.startswith("hasil tersimpan, ") for c in captions(at)) and meta.stat().st_mtime_ns == stamp
    # Run baru (walau dipakai ulang) menjadi aktif, dan pemilih ikut pindah dari OPT lama ke K=5.
    assert at.session_state["active_run"] == K5_CPU_ID
    assert at.selectbox(key="active_run_pick").value == K5_CPU_ID
    time.sleep(0.05)                                   # supaya waktu ubah berkas pasti berbeda
    at.sidebar.slider(key="f_k").set_value(5)
    at.sidebar.checkbox(key="f_force").check()
    run_mode(at, "cpu")
    no_errors(at)
    assert meta.stat().st_mtime_ns > stamp
    assert not any(c.startswith("hasil tersimpan, ") for c in captions(at))


def test_class_map_dibuat_per_run_dan_kosong(sandbox):
    from landcover.labeling import UNNAMED

    at = start()
    run_mode(at, "cpu")
    assert at.session_state["class_maps"] == {CPU_ID: {i: UNNAMED for i in range(8)}}
    at.sidebar.slider(key="f_k").set_value(3)
    run_mode(at, "cpu")
    maps = at.session_state["class_maps"]
    assert set(maps) == {CPU_ID, "kecil__cpu__k3_it20_s42_t1"}
    assert len(maps["kecil__cpu__k3_it20_s42_t1"]) == 3


def test_dataset_sintetis_dan_citra_asli(sandbox):
    at = start()
    at.sidebar.selectbox(key="f_dataset").select("tiruan_sleman")
    run_mode(at, "cpu")
    no_errors(at)
    assert at.session_state["active_run"].startswith("tiruan_sleman__cpu__")
    assert len(at.get("image")) == 2                                      # citra sintetis dibuat ulang
    assert not any("tidak ditemukan" in i.value for i in at.info)


def test_opsi_lanjutan_dipakai(sandbox):
    at = start()
    at.sidebar.selectbox(key="f_kmeans").select("omp")
    at.sidebar.slider(key="f_threads").set_value(4)
    at.sidebar.number_input(key="f_seed").set_value(7)
    at.sidebar.number_input(key="f_iters").set_value(5)
    run_mode(at, "opt")
    no_errors(at)
    assert at.session_state["active_run"] == "kecil__opt-omp__k8_it5_s7_t4"


# --- waktu berjalan ------------------------------------------------------------ #

def test_run_dengan_tidur_selesai_dan_timer_tidak_merusak_halaman(sandbox, monkeypatch):
    monkeypatch.setenv("LC_MOCK_SLEEP", "0.5")
    at = start()
    t0 = time.perf_counter()
    run_mode(at, "cpu")
    assert time.perf_counter() - t0 >= 0.5
    no_errors(at)
    assert metrics(at)["Total: CPU saja (L0)"] != "—"
    assert not any("berjalan:" in m.value for m in at.markdown)              # timer dibersihkan


# --- galat dan keadaan tepi ----------------------------------------------------- #

def test_backend_cli_halaman_tampil_dan_jalankan_menampilkan_galat(sandbox, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BACKEND", "cli")
    monkeypatch.setattr(config, "CORE_BIN", tmp_path / "tidak_ada" / "landcover")   # biner tidak ada (hermetik)
    at = start()
    assert at.sidebar.selectbox(key="f_dataset").options == ["kecil (PNG)"]     # tanpa dataset sintetis
    assert any("Program landcover belum bisa dipakai" in w.value for w in at.warning)
    press(at, "btn_run")
    assert len(at.error) == 1 and "Program landcover tidak ditemukan" in at.error[0].value
    assert at.session_state["runs"] == {}
    assert not any("DATA TIRUAN" in m.value for m in at.markdown)
    assert not any("DATA TIRUAN" in w.value for w in at.warning)
    press(at, "btn_compare")
    assert len(at.error) == 1 and at.session_state["runs"] == {}


def test_backend_cli_tanpa_dataset_tombol_mati(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "cli")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "kosong")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    at = start()
    assert any("Tidak ada dataset" in w.value for w in at.sidebar.warning)
    assert at.sidebar.button(key="btn_run").disabled and at.sidebar.button(key="btn_compare").disabled


def test_galat_launcher_tampil_sebagai_pesan(sandbox, monkeypatch):
    from launchers.mock import MockLauncher

    def boom(self, params, out_dir, on_progress=None):
        raise OSError("disk penuh")

    at = start()
    monkeypatch.setattr(MockLauncher, "run", boom)
    press(at, "btn_run")
    assert len(at.error) == 1 and "gagal" in at.error[0].value and "disk penuh" in at.error[0].value
    assert at.session_state["runs"] == {}


def test_galat_pada_run_kedua_bandingkan_mempertahankan_run_pertama(sandbox, monkeypatch):
    from launchers.mock import MockLauncher

    real = MockLauncher.run

    def hanya_cpu(self, params, out_dir, on_progress=None):
        if params.mode == "opt":
            raise RuntimeError("GPU tidak ada")
        return real(self, params, out_dir, on_progress)

    at = start()
    monkeypatch.setattr(MockLauncher, "run", hanya_cpu)
    press(at, "btn_compare")
    assert len(at.error) == 1 and "GPU tidak ada" in at.error[0].value
    assert list(at.session_state["runs"]) == [CPU_ID]
    assert metrics(at)["Total: CPU saja (L0)"] != "—"                       # hasil CPU tetap tampil


def test_folder_run_hilang_tidak_membuat_halaman_mogok(sandbox):
    import shutil

    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    shutil.rmtree(runs)
    at.run()
    assert not at.exception
    assert any("tidak terbaca" in w.value for w in at.warning)
    assert any("Semua hasil" in e.value for e in at.error)


def test_dataset_terpilih_dihapus_dari_disk_tidak_mogok(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    (data / "kecil.png").unlink()
    at.run()
    assert not at.exception
    assert "kecil (PNG)" not in at.sidebar.selectbox(key="f_dataset").options
    assert metrics(at)["Total: CPU saja (L0)"] != "—"                       # hasil lama tetap ada


def test_hasil_janggal_ditampilkan_dengan_peringatan(sandbox):
    import json

    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    counts = runs / "mock" / CPU_ID / "counts.json"
    values = json.loads(counts.read_text(encoding="utf-8"))
    values[0] += 5                                                          # jumlah != W*H
    counts.write_text(json.dumps(values), encoding="utf-8")
    meta = runs / "mock" / CPU_ID / "meta.json"
    meta.touch()                                                            # waktu ubah baru: cache dibaca ulang
    at.run()
    assert not at.exception
    assert any("janggal" in w.value for w in at.warning)
    assert metrics(at)["Total: CPU saja (L0)"] != "—"


def test_value_error_dari_parameter_tampil_sebagai_pesan(sandbox, monkeypatch):
    """Parameter yang ditolak RunParams (ValueError) tidak boleh membuat halaman mogok."""
    import ui.sidebar as sidebar

    def gagal(*args, **kwargs):
        raise ValueError("nama dataset tidak aman")

    at = start()
    monkeypatch.setattr(sidebar, "build_params", gagal)     # skrip mengimpor ulang nama ini tiap eksekusi
    press(at, "btn_run")
    assert [e.value for e in at.error] == ["nama dataset tidak aman"]
    assert at.session_state["runs"] == {}


# --- grafik waktu per tahap (Langkah 7) ---------------------------------------- #

def charts_of(at: AppTest) -> list[dict]:
    """Spesifikasi plotly (dict) tiap elemen `plotly_chart`, ditambah kunci widgetnya."""
    out = []
    for el in at.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        spec["_key"] = el.key
        out.append(spec)
    return out


def bars_y(spec: dict) -> list[str]:
    stage = [t for t in spec["data"] if t.get("meta", {}).get("kind") == "stage"]
    return list(stage[0]["y"])


def total_trace(spec: dict) -> dict:
    (t,) = [t for t in spec["data"] if t.get("meta", {}).get("kind") == "total"]
    return t


def test_grafik_waktu_tampil_untuk_satu_dan_dua_run_dan_ikut_run_aktif(sandbox):
    at = start()
    assert charts_of(at) == []                                            # belum ada hasil: hanya kotak kosong
    assert any("Grafik waktu per tahap: belum ada hasil" in c for c in captions(at))

    run_mode(at, "cpu")                                                   # satu run
    no_errors(at)
    (spec,) = charts_of(at)
    assert spec["_key"] == "chart_stage_seconds" and bars_y(spec) == ["CPU saja (L0)"]
    assert "DATA TIRUAN" in spec["layout"]["title"]["text"]               # lencana di dalam grafik
    assert any(s.value.startswith("Waktu per tahap") and "DATA TIRUAN" in s.value for s in at.subheader)
    assert not any("Grafik waktu per tahap: belum ada hasil" in c for c in captions(at))
    cpu_total = float(total_trace(spec)["x"][0])
    assert cpu_total > 0

    run_mode(at, "opt")                                                   # dua run
    no_errors(at)
    (spec,) = charts_of(at)
    assert bars_y(spec) == ["CPU saja (L0)", "Optimalisasi · OpenMP"]     # dataset kecil: auto memilih OpenMP
    assert float(total_trace(spec)["x"][0]) == pytest.approx(cpu_total)   # total = timing.json, bukan dihitung ulang
    assert float(total_trace(spec)["x"][1]) < cpu_total
    devices = [t["meta"]["device"] for t in spec["data"] if t.get("meta", {}).get("kind") == "legend"]
    assert devices == ["serial", "omp"]

    # K lain (run CPU baru tanpa pasangan) -> grafik satu bar; memilih run lama -> dua bar lagi
    at.sidebar.slider(key="f_k").set_value(5)
    run_mode(at, "cpu")
    no_errors(at)
    assert [bars_y(s) for s in charts_of(at)] == [["CPU saja (L0)"]]
    at.selectbox(key="active_run_pick").select_index(1).run()
    assert not at.exception
    assert at.session_state["active_run"] == OPT_ID
    assert [bars_y(s) for s in charts_of(at)] == [["CPU saja (L0)", "Optimalisasi · OpenMP"]]


def test_toggle_persentase_tidak_menghilangkan_hasil(sandbox):
    at = start()
    run_mode(at, "cpu")
    run_mode(at, "opt")
    shown = metrics(at)
    assert at.radio(key="chart_view").value == "seconds"

    at.radio(key="chart_view").set_value("percent").run()
    assert not at.exception
    (spec,) = charts_of(at)
    assert spec["_key"] == "chart_stage_percent"
    assert [float(v) for v in total_trace(spec)["x"]] == [100.0, 100.0]  # sumbu persen: total hanya teks di ujung bar
    stage = [t for t in spec["data"] if t.get("meta", {}).get("kind") == "stage"]
    for i in range(2):                                                    # tiap bar berjumlah 100 %
        assert sum(float(t["x"][i]) for t in stage) == pytest.approx(100.0)
    assert metrics(at) == shown and len(at.get("image")) == 2             # kartu dan peta tetap
    assert list(at.session_state["runs"]) == [CPU_ID, OPT_ID]

    at.radio(key="chart_view").set_value("seconds").run()
    assert not at.exception and charts_of(at)[0]["_key"] == "chart_stage_seconds"


def test_tabel_waktu_per_tahap_ada_di_expander(sandbox):
    at = start()
    run_mode(at, "cpu")
    run_mode(at, "opt")
    assert any(e.label == "Tabel waktu per tahap" for e in at.expander)
    (frame,) = [f for f in at.dataframe if "Tahap" in f.value.columns]    # kembaran tabel grafik
    df = frame.value                                                      # (panel pelabelan punya tabelnya sendiri)
    assert list(df["Run"].unique()) == ["CPU saja (L0)", "Optimalisasi · OpenMP"]
    assert len(df) == 2 * 6                                               # 5 tahap + total resmi per run
    totals = df[df["Tahap"] == "Total resmi (timing.json)"]["Waktu (s)"].tolist()
    assert totals[0] > totals[1] > 0                                      # CPU saja lebih lambat dari Optimalisasi
