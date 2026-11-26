"""State sesi, hasil CPU tersimpan, dan eksekusi run dengan timer (logika murni, tanpa Streamlit).

Diekstrak dari `ui/runs.py` (Fase 1 UI web): dipakai bersama oleh aplikasi Streamlit (`ui/runs.py`
hanya menambah cache `st.cache_data`) dan API FastAPI (`api/`). Hampir semuanya fungsi murni:
`state` adalah `st.session_state` atau dict biasa.

Kunci state (nama dari plan; Langkah 7-9 bergantung padanya):
    runs         run_id -> path folder run (str); urutan penyisipan = urutan terbaru di akhir
    active_run   run_id yang ditampilkan (atau tidak ada)
    class_maps   run_id -> ClassMap (Langkah 8 yang mengubah isinya)
    run_notes    run_id -> catatan teks, mis. "hasil tersimpan, 21-11-2026 10:15"
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, MutableMapping

from landcover.contract import ContractError, RunResult, read_run, validate
from landcover.labeling import ClassMap, default_class_map
from launchers.base import Launcher, RunParams, out_dir_for, run_id

RUNS_KEY = "runs"
ACTIVE_KEY = "active_run"
CLASS_MAPS_KEY = "class_maps"
NOTES_KEY = "run_notes"

MODE_LABELS = {"cpu": "CPU saja (L0)", "opt": "Optimalisasi"}
KMEANS_LABELS = {"auto": "Otomatis", "omp": "OpenMP", "cuda": "CUDA"}
DEVICE_LABELS = {"serial": "CPU serial", "omp": "OpenMP", "cuda": "CUDA"}

TICK_SECONDS = 0.1


# --- state ---------------------------------------------------------------- #

def ensure_state(state: MutableMapping) -> None:
    for key in (RUNS_KEY, CLASS_MAPS_KEY, NOTES_KEY):
        state.setdefault(key, {})


def register_run(state: MutableMapping, rid: str, path: Path | str, note: str | None = None) -> None:
    """Catat run di sesi dan jadikan aktif. Run yang sama dipindah ke urutan terbaru."""
    ensure_state(state)
    state[RUNS_KEY].pop(rid, None)
    state[RUNS_KEY][rid] = str(path)
    if note:
        state[NOTES_KEY][rid] = note
    else:
        state[NOTES_KEY].pop(rid, None)
    state[ACTIVE_KEY] = rid


def class_map_for(state: MutableMapping, rid: str, k: int) -> ClassMap:
    """ClassMap run `rid` (dibuat kosong, semua cluster tanpa nama, pada akses pertama).

    Disimpan per run_id (yang memuat K), jadi nama kelas dari run lain tidak ikut."""
    ensure_state(state)
    maps = state[CLASS_MAPS_KEY]
    if rid not in maps:
        maps[rid] = default_class_map(k)
    return maps[rid]


def run_id_of(run: RunResult) -> str:
    """Nama folder run = run_id (`out_dir_for` menulisnya begitu)."""
    return run.path.name


# --- membaca folder run --------------------------------------------------- #

def run_stamp(path: Path | str) -> int:
    """Waktu ubah `meta.json` (ns). Ditulis terakhir oleh launcher, jadi cukup sebagai kunci cache."""
    try:
        return (Path(path) / "meta.json").stat().st_mtime_ns
    except OSError:
        return 0


@lru_cache(maxsize=64)
def _read_cached(path: str, stamp: int) -> RunResult:   # noqa: ARG001 (stamp = kunci cache)
    return read_run(path)


def load_run(path: Path | str) -> RunResult:
    """`read_run` dengan cache berkunci (path, waktu ubah `meta.json`). ContractError bila rusak.

    Cache ini untuk API; aplikasi Streamlit memakai `ui.runs.load_run` (cache `st.cache_data`)."""
    return _read_cached(str(path), run_stamp(path))


def load_session_runs(
    state: MutableMapping,
    loader: Callable[[Path | str], RunResult] | None = None,
) -> tuple[dict[str, RunResult], dict[str, str]]:
    """Baca semua run sesi: (yang terbaca, {run_id: alasan gagal}). Urutan = urutan state.

    `loader` mengganti pembaca (Streamlit memberi versi ber-cache `st.cache_data`)."""
    ensure_state(state)
    read = loader or load_run
    loaded: dict[str, RunResult] = {}
    failed: dict[str, str] = {}
    for rid, path in state[RUNS_KEY].items():
        try:
            loaded[rid] = read(path)
        except (ContractError, OSError) as exc:
            failed[rid] = str(exc)
    return loaded, failed


def created_text(meta: dict, path: Path | str) -> str:
    """Tanggal-jam run untuk tampilan: dari `created_at`, cadangan waktu ubah meta.json."""
    try:
        moment = datetime.fromisoformat(str(meta["created_at"]))
    except (KeyError, ValueError):
        try:
            moment = datetime.fromtimestamp((Path(path) / "meta.json").stat().st_mtime)
        except OSError:
            return "waktu tidak diketahui"
    return moment.strftime("%d-%m-%Y %H:%M")


def describe_run(run: RunResult, note: str | None = None) -> str:
    """Satu baris untuk pemilih hasil: dataset, mode, K, iterasi, seed (+ catatan)."""
    m = run.meta
    if run.mode == "cpu":
        engine = MODE_LABELS["cpu"]
    else:
        engine = f"{MODE_LABELS['opt']} ({DEVICE_LABELS.get(run.kmeans_device, run.kmeans_device)})"
    text = f"{m.get('dataset', '?')} · {engine} · K={run.k} · {m.get('iters', '?')} iter · seed {m.get('seed', '?')}"
    return f"{text} · {note}" if note else text


# --- hasil CPU tersimpan -------------------------------------------------- #

def find_saved_run(runs_dir: Path | str, backend: str, params: RunParams) -> Path | None:
    """Folder run CPU saja yang sudah ada dan boleh dipakai ulang, atau None.

    Syarat: mode cpu; folder terbaca; `validate()` kosong; meta cocok dengan parameter
    (dataset, k, iters, seed); `mock` sesuai backend. Selain itu dianggap tidak ada dan
    run baru menimpa folder itu. Isi dataset TIDAK diperiksa: bila citra diganti, pakai
    "paksa jalankan ulang"."""
    if params.mode != "cpu":
        return None
    out = out_dir_for(runs_dir, backend, params)
    if not (out / "meta.json").is_file():
        return None
    try:
        run = read_run(out)
    except (ContractError, OSError):
        return None
    if validate(run):
        return None
    m = run.meta
    if (m.get("dataset"), m.get("mode"), m.get("k"), m.get("iters"), m.get("seed")) != (
        params.dataset, "cpu", params.k, params.iters, params.seed,
    ):
        return None
    if run.mock != (backend == "mock"):
        return None
    return out


# --- eksekusi dengan timer ------------------------------------------------ #

_JOBS: dict[str, threading.Thread] = {}
_JOBS_LOCK = threading.Lock()


def execute(
    launcher: Launcher,
    params: RunParams,
    out_dir: Path,
    *,
    on_tick: Callable[[float], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    tick: float = TICK_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Jalankan `launcher.run` di thread; thread pemanggil memperbarui timer tiap `tick` detik.

    `on_tick(detik_berjalan)` dan `on_progress(baris)` selalu dipanggil dari thread
    pemanggil (aman untuk `st.*`); thread pekerja tidak pernah menyentuh Streamlit.
    `on_tick` dipanggil sekali lagi di akhir dengan waktu total. Galat launcher dilempar
    ulang di sini. Bila skrip Streamlit terhenti di tengah jalan, thread pekerja tetap
    selesai; run berikutnya ke folder yang sama menunggunya dulu supaya tidak ada dua
    penulis di satu folder."""
    key = str(Path(out_dir).resolve())
    with _JOBS_LOCK:
        previous = _JOBS.get(key)
    if previous is not None and previous.is_alive():
        previous.join()

    lines: queue.SimpleQueue[str] = queue.SimpleQueue()
    outcome: dict[str, object] = {}

    def work() -> None:
        try:
            outcome["path"] = launcher.run(params, out_dir, on_progress=lines.put)
        except BaseException as exc:      # dilempar ulang di thread pemanggil
            outcome["error"] = exc

    def drain() -> None:
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            if on_progress:
                on_progress(line)

    worker = threading.Thread(target=work, name=f"run-{run_id(params)}", daemon=True)
    with _JOBS_LOCK:
        _JOBS[key] = worker
    started = clock()
    worker.start()
    while worker.is_alive():
        drain()
        if on_tick:
            on_tick(clock() - started)
        sleep(tick)
    worker.join()
    drain()
    if on_tick:
        on_tick(clock() - started)
    with _JOBS_LOCK:
        if _JOBS.get(key) is worker:
            del _JOBS[key]
    if "error" in outcome:
        raise outcome["error"]      # type: ignore[misc]
    return outcome["path"]          # type: ignore[return-value]


@dataclass(frozen=True)
class Outcome:
    run_id: str
    path: Path
    reused: bool
    note: str | None        # "hasil tersimpan, <tanggal jam>" bila dipakai ulang


def obtain_run(
    launcher: Launcher,
    runs_dir: Path | str,
    backend: str,
    params: RunParams,
    *,
    force: bool = False,
    on_tick: Callable[[float], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    tick: float = TICK_SECONDS,
) -> Outcome:
    """Pakai hasil CPU tersimpan bila ada (dan tidak `force`), selain itu jalankan launcher."""
    rid = run_id(params)
    if not force:
        saved = find_saved_run(runs_dir, backend, params)
        if saved is not None:
            meta = read_run(saved).meta
            return Outcome(rid, saved, True, f"hasil tersimpan, {created_text(meta, saved)}")
    path = execute(
        launcher, params, out_dir_for(runs_dir, backend, params),
        on_tick=on_tick, on_progress=on_progress, tick=tick,
    )
    return Outcome(rid, Path(path), False, None)


__all__ = [
    "ACTIVE_KEY", "CLASS_MAPS_KEY", "DEVICE_LABELS", "KMEANS_LABELS", "MODE_LABELS", "NOTES_KEY",
    "Outcome", "RUNS_KEY", "TICK_SECONDS", "class_map_for", "created_text", "describe_run",
    "ensure_state", "execute", "find_saved_run", "load_run", "load_session_runs", "obtain_run",
    "register_run", "run_id_of", "run_stamp",
]
