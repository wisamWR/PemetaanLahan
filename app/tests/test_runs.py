"""State sesi, hasil CPU tersimpan, dan eksekusi bertimer (ui/runs.py, Langkah 6)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import write_run_dir
from landcover.contract import ContractError, read_run
from launchers.base import RunParams, out_dir_for, run_id
from launchers.mock import MockLauncher
from ui import runs as R


def params(**kw) -> RunParams:
    base = dict(dataset="kecil", mode="cpu", kmeans="auto", k=4, iters=3, seed=42, threads=8)
    base.update(kw)
    return RunParams(**base)


@pytest.fixture
def env(tmp_path):
    """data/ berisi satu PNG kecil acak (K-Means tiruan cepat); runs/ kosong."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(data / "kecil.png")
    return data, runs


class SpyLauncher:
    """Membungkus MockLauncher dan menghitung berapa kali benar-benar dijalankan."""

    def __init__(self, data):
        self.inner, self.calls = MockLauncher(data), []

    def run(self, p, out_dir, on_progress=None):
        self.calls.append(run_id(p))
        return self.inner.run(p, out_dir, on_progress)


# --- state ----------------------------------------------------------------- #

def test_register_run_urutan_terbaru_di_akhir_dan_menjadi_aktif():
    state: dict = {}
    R.register_run(state, "a", "/p/a")
    R.register_run(state, "b", "/p/b", "hasil tersimpan, 21-11-2026 10:15")
    R.register_run(state, "a", "/p/a2")                       # run ulang: pindah ke akhir
    assert list(state[R.RUNS_KEY]) == ["b", "a"]
    assert state[R.RUNS_KEY]["a"] == "/p/a2"
    assert state[R.ACTIVE_KEY] == "a"
    assert state[R.NOTES_KEY] == {"b": "hasil tersimpan, 21-11-2026 10:15"}


def test_register_run_tanpa_catatan_menghapus_catatan_lama():
    state: dict = {}
    R.register_run(state, "a", "/p/a", "hasil tersimpan, x")
    R.register_run(state, "a", "/p/a")                        # sekarang hasil baru, bukan tersimpan
    assert "a" not in state[R.NOTES_KEY]


def test_kunci_state_sesuai_plan():
    assert (R.RUNS_KEY, R.ACTIVE_KEY, R.CLASS_MAPS_KEY) == ("runs", "active_run", "class_maps")


def test_class_map_per_run_id_dan_awalnya_tanpa_nama():
    from landcover.labeling import UNNAMED

    state: dict = {}
    a = R.class_map_for(state, "run_a", 3)
    assert a == {0: UNNAMED, 1: UNNAMED, 2: UNNAMED}
    a[1] = "Air"
    assert R.class_map_for(state, "run_a", 3)[1] == "Air"        # objek yang sama, bertahan
    assert R.class_map_for(state, "run_b", 2) == {0: UNNAMED, 1: UNNAMED}   # tidak bocor antar run
    assert state[R.CLASS_MAPS_KEY]["run_a"][1] == "Air"


# --- membaca folder run ---------------------------------------------------- #

def test_load_run_cache_berkunci_waktu_ubah(tmp_path):
    folder = write_run_dir(tmp_path / "r", k=4)
    assert R.load_run(folder).k == 4
    assert R.load_run(folder).k == 4                            # kena cache, hasil sama
    write_run_dir(folder, k=3)
    stamp = R.run_stamp(folder)
    os.utime(folder / "meta.json", ns=(stamp + 10**9, stamp + 10**9))   # waktu ubah pasti berbeda
    assert R.load_run(folder).k == 3                            # cache tidak basi


def test_load_run_folder_tanpa_meta_melempar_contract_error(tmp_path):
    folder = write_run_dir(tmp_path / "r", drop=("meta.json",))
    assert R.run_stamp(folder) == 0
    with pytest.raises(ContractError):
        R.load_run(folder)


def test_load_session_runs_memisahkan_yang_gagal(tmp_path):
    good = write_run_dir(tmp_path / "ok")
    state: dict = {}
    R.register_run(state, "ok", good)
    R.register_run(state, "hilang", tmp_path / "tidak_ada")
    loaded, failed = R.load_session_runs(state)
    assert list(loaded) == ["ok"] and list(failed) == ["hilang"]
    assert failed["hilang"]                                      # ada alasan


def test_created_text(tmp_path):
    assert R.created_text({"created_at": "2026-11-21T10:15:00+07:00"}, tmp_path) == "21-11-2026 10:15"
    (tmp_path / "meta.json").write_text("{}")
    assert re.fullmatch(r"\d\d-\d\d-\d{4} \d\d:\d\d", R.created_text({"created_at": "bukan tanggal"}, tmp_path))
    assert re.fullmatch(r"\d\d-\d\d-\d{4} \d\d:\d\d", R.created_text({}, tmp_path))
    assert R.created_text({}, tmp_path / "tidak_ada") == "waktu tidak diketahui"


def test_describe_run(tmp_path):
    cpu = read_run(write_run_dir(tmp_path / "c", meta={"mode": "cpu", "kmeans_device": "serial",
                                                       "threads": 1, "dataset": "sleman"}))
    opt = read_run(write_run_dir(tmp_path / "o", meta={"dataset": "sleman"}))
    assert R.describe_run(cpu) == "sleman · CPU saja (L0) · K=4 · 20 iter · seed 42"
    assert R.describe_run(opt, "hasil tersimpan, 21-11-2026 10:15") == (
        "sleman · Optimalisasi (CUDA) · K=4 · 20 iter · seed 42 · hasil tersimpan, 21-11-2026 10:15"
    )


# --- hasil CPU tersimpan --------------------------------------------------- #

def jalankan(env, p, backend="mock"):
    data, runs = env
    return MockLauncher(data).run(p, out_dir_for(runs, backend, p))


def test_hasil_cpu_yang_valid_dipakai_ulang(env):
    _, runs = env
    p = params()
    out = jalankan(env, p)
    assert R.find_saved_run(runs, "mock", p) == out
    # kmeans dan thread tidak memengaruhi run_id mode cpu, jadi tetap ketemu
    assert R.find_saved_run(runs, "mock", params(kmeans="cuda", threads=2)) == out


def test_hasil_opt_tidak_pernah_dipakai_ulang(env):
    _, runs = env
    p = params(mode="opt", kmeans="omp")
    jalankan(env, p)
    assert R.find_saved_run(runs, "mock", p) is None


def test_hasil_tersimpan_tidak_ada_atau_parameter_beda(env):
    _, runs = env
    assert R.find_saved_run(runs, "mock", params()) is None
    jalankan(env, params())
    for kw in ({"k": 5}, {"iters": 4}, {"seed": 7}, {"dataset": "lain"}):
        assert R.find_saved_run(runs, "mock", params(**kw)) is None, kw


def test_hasil_tersimpan_ditolak_bila_meta_tidak_cocok(env):
    _, runs = env
    p = params()
    out = jalankan(env, p)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    for key, value in (("seed", 99), ("iters", 9), ("k", 5), ("dataset", "x"), ("mode", "opt")):
        broken = dict(meta, **{key: value})
        (out / "meta.json").write_text(json.dumps(broken), encoding="utf-8")
        assert R.find_saved_run(runs, "mock", p) is None, key


def test_hasil_tersimpan_ditolak_bila_rusak_atau_janggal(env):
    _, runs = env
    p = params()
    out = jalankan(env, p)
    counts = (out / "counts.json").read_text(encoding="utf-8")
    (out / "counts.json").write_text("[1, 2", encoding="utf-8")                 # JSON rusak
    assert R.find_saved_run(runs, "mock", p) is None
    (out / "counts.json").write_text(json.dumps([1, 1, 1, 1]), encoding="utf-8")   # jumlah != W*H
    assert R.find_saved_run(runs, "mock", p) is None
    (out / "counts.json").write_text(counts, encoding="utf-8")
    assert R.find_saved_run(runs, "mock", p) == out                             # pulih
    (out / "labels_preview.png").unlink()                                        # folder tak lengkap
    assert R.find_saved_run(runs, "mock", p) is None


def test_hasil_tersimpan_ditolak_bila_mock_tidak_sesuai_backend(env):
    _, runs = env
    p = params()
    out = jalankan(env, p, backend="cli")        # hasil tiruan "menyamar" di runs/core/
    assert read_run(out).mock is True
    assert R.find_saved_run(runs, "cli", p) is None
    assert R.find_saved_run(runs, "mock", p) is None    # dan runs/mock/ memang kosong


def test_obtain_run_memakai_ulang_cpu_tetapi_selalu_menjalankan_opt(env):
    data, runs = env
    spy = SpyLauncher(data)

    first = R.obtain_run(spy, runs, "mock", params())
    assert (first.reused, first.note) == (False, None)
    second = R.obtain_run(spy, runs, "mock", params())
    assert second.reused and second.path == first.path
    assert re.fullmatch(r"hasil tersimpan, \d\d-\d\d-\d{4} \d\d:\d\d", second.note)
    assert len(spy.calls) == 1

    opt = params(mode="opt", kmeans="omp")
    R.obtain_run(spy, runs, "mock", opt)
    again = R.obtain_run(spy, runs, "mock", opt)
    assert not again.reused and len(spy.calls) == 3          # opt dijalankan tiap kali

    forced = R.obtain_run(spy, runs, "mock", params(), force=True)
    assert not forced.reused and forced.note is None and len(spy.calls) == 4
    assert forced.run_id == run_id(params())


def test_obtain_run_menimpa_folder_cpu_yang_rusak(env):
    data, runs = env
    spy = SpyLauncher(data)
    p = params()
    out = R.obtain_run(spy, runs, "mock", p).path
    (out / "timing.json").write_text("{", encoding="utf-8")
    redo = R.obtain_run(spy, runs, "mock", p)
    assert not redo.reused and len(spy.calls) == 2
    assert read_run(redo.path).total > 0


# --- execute: thread, timer, progres --------------------------------------- #

class FakeLauncher:
    def __init__(self, *, lines=(), hold: threading.Event | None = None, error: Exception | None = None):
        self.lines, self.hold, self.error = lines, hold, error
        self.thread_id = None
        self.started = threading.Event()
        self.finished = threading.Event()

    def run(self, p, out_dir, on_progress=None):
        self.thread_id = threading.get_ident()
        self.started.set()
        for line in self.lines:
            on_progress(line)
        if self.hold is not None:
            assert self.hold.wait(10), "tes macet: hold tidak pernah dilepas"
        try:
            if self.error:
                raise self.error
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return Path(out_dir)
        finally:
            self.finished.set()


def test_execute_timer_berdetak_dan_callback_di_thread_pemanggil(tmp_path):
    hold = threading.Event()
    launcher = FakeLauncher(lines=("[stage] baca selesai 0.10 s", "[stage] tulis selesai 0.20 s"), hold=hold)
    sleeps: list[float] = []
    ticks: list[float] = []
    progress: list[str] = []
    callers: set[int] = set()
    now = [0.0]

    def fake_clock():
        return now[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if len(sleeps) == 3:
            hold.set()                    # lepaskan pekerja sesudah tiga detak

    def on_tick(s):
        callers.add(threading.get_ident())
        ticks.append(s)

    def on_progress(line):
        callers.add(threading.get_ident())
        progress.append(line)

    out = R.execute(launcher, params(), tmp_path / "out", on_tick=on_tick, on_progress=on_progress,
                    sleep=fake_sleep, clock=fake_clock)
    assert out == tmp_path / "out"
    assert sleeps and all(s == R.TICK_SECONDS == 0.1 for s in sleeps)
    assert len(ticks) >= 4 and ticks == sorted(ticks) and ticks[0] == 0.0 and ticks[-1] >= 0.3
    assert progress == ["[stage] baca selesai 0.10 s", "[stage] tulis selesai 0.20 s"]
    assert callers == {threading.get_ident()}                 # st.* aman: tidak dari thread pekerja
    assert launcher.thread_id != threading.get_ident()        # launcher jalan di thread lain


def test_execute_tanpa_callback_dan_pekerja_cepat(tmp_path):
    assert R.execute(FakeLauncher(), params(), tmp_path / "o") == tmp_path / "o"


def test_execute_tick_terakhir_membawa_waktu_total(tmp_path):
    """Tick penutup melaporkan jam saat selesai. Tick di dalam loop selalu dihitung SEBELUM
    sleep, jadi hanya tick penutup yang bisa sama dengan jam akhir (jam palsu maju 0,1 detik
    per sleep). Jumlah putaran loop tidak ikut diasumsikan: thread yang baru selesai bisa
    sesaat masih `is_alive()` sehingga loop berputar sekali-dua kali lagi."""
    hold = threading.Event()
    launcher = FakeLauncher(hold=hold)
    now = [0.0]
    ticks: list[float] = []

    def fake_sleep(seconds):
        now[0] += seconds
        hold.set()
        assert launcher.finished.wait(5)

    R.execute(launcher, params(), tmp_path / "o", on_tick=ticks.append, sleep=fake_sleep, clock=lambda: now[0])
    assert len(ticks) >= 2 and ticks[0] == 0.0 and ticks == sorted(ticks)
    assert now[0] >= 0.1 and ticks[-1] == now[0]


def test_execute_melempar_ulang_galat_launcher(tmp_path):
    launcher = FakeLauncher(error=ValueError("dataset rusak"))
    with pytest.raises(ValueError, match="dataset rusak"):
        R.execute(launcher, params(), tmp_path / "o")
    # tidak ada thread yatim tercatat, run berikutnya ke folder sama langsung jalan
    assert R.execute(FakeLauncher(), params(), tmp_path / "o") == tmp_path / "o"


def test_execute_run_kedua_ke_folder_sama_menunggu_thread_yatim(tmp_path):
    """Skrip Streamlit terhenti di tengah run (on_tick melempar): pekerja tetap hidup.
    Run berikutnya ke folder yang sama tidak boleh mulai sebelum pekerja itu selesai."""
    class Stop(Exception):
        pass

    hold = threading.Event()
    first = FakeLauncher(hold=hold)
    out = tmp_path / "o"

    def abort(_s):
        raise Stop

    with pytest.raises(Stop):
        R.execute(first, params(), out, on_tick=abort)
    assert first.started.wait(5) and not first.finished.is_set()      # yatim: masih berjalan

    second = FakeLauncher()
    done = threading.Event()

    def again():
        R.execute(second, params(), out)
        done.set()

    t = threading.Thread(target=again)
    t.start()
    time.sleep(0.3)
    assert not second.started.is_set()                     # menunggu, belum menyentuh folder
    hold.set()
    assert done.wait(10)
    t.join()
    assert first.finished.is_set() and second.started.is_set()


def test_execute_folder_berbeda_tidak_saling_menunggu(tmp_path):
    hold = threading.Event()
    blocked = FakeLauncher(hold=hold)

    class Stop(Exception):
        pass

    def abort(_s):
        raise Stop

    with pytest.raises(Stop):
        R.execute(blocked, params(), tmp_path / "a", on_tick=abort)
    try:
        assert R.execute(FakeLauncher(), params(), tmp_path / "b") == tmp_path / "b"
    finally:
        hold.set()


def test_execute_dengan_mock_launcher_nyata_menyampaikan_progres(env):
    data, runs = env
    lines: list[str] = []
    p = params()
    out = R.execute(MockLauncher(data, sleep=0.0), p, out_dir_for(runs, "mock", p), on_progress=lines.append)
    assert read_run(out).k == 4
    stages = [ln for ln in lines if ln.startswith("[stage]")]
    assert [ln.split()[1] for ln in stages] == ["baca", "inisialisasi", "kmeans", "pewarnaan", "tulis"]
