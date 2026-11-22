"""State sesi, hasil CPU tersimpan, dan eksekusi run dengan timer (plan Langkah 6).

Logika murni ada di `services/runs.py` (dipakai bersama API FastAPI); modul ini mengimpornya dan
hanya menambah cache Streamlit (`st.cache_data`) pada `load_run`, jadi nama-nama lama
(`ui.runs.execute`, `ui.runs.register_run`, dst.) tetap berlaku.

Kunci state (nama dari plan; Langkah 7-9 bergantung padanya):
    runs         run_id -> path folder run (str); urutan penyisipan = urutan terbaru di akhir
    active_run   run_id yang ditampilkan (atau tidak ada)
    class_maps   run_id -> ClassMap (Langkah 8 yang mengubah isinya)
    run_notes    run_id -> catatan teks, mis. "hasil tersimpan, 21-11-2026 10:15"
"""

from __future__ import annotations

from pathlib import Path
from typing import MutableMapping

import streamlit as st

from landcover.contract import RunResult, read_run
from services import runs as _svc
from services.runs import (
    ACTIVE_KEY,
    CLASS_MAPS_KEY,
    DEVICE_LABELS,
    KMEANS_LABELS,
    MODE_LABELS,
    NOTES_KEY,
    RUNS_KEY,
    TICK_SECONDS,
    Outcome,
    class_map_for,
    created_text,
    describe_run,
    ensure_state,
    execute,
    find_saved_run,
    obtain_run,
    register_run,
    run_id_of,
    run_stamp,
)


@st.cache_data(show_spinner=False, max_entries=64)
def _read_cached(path: str, stamp: int) -> RunResult:   # noqa: ARG001 (stamp = kunci cache)
    return read_run(path)


def load_run(path: Path | str) -> RunResult:
    """`read_run` dengan cache berkunci (path, waktu ubah). ContractError bila rusak."""
    return _read_cached(str(path), run_stamp(path))


def load_session_runs(state: MutableMapping) -> tuple[dict[str, RunResult], dict[str, str]]:
    """Baca semua run sesi: (yang terbaca, {run_id: alasan gagal}). Urutan = urutan state."""
    return _svc.load_session_runs(state, loader=load_run)


__all__ = [
    "ACTIVE_KEY", "CLASS_MAPS_KEY", "DEVICE_LABELS", "KMEANS_LABELS", "MODE_LABELS", "NOTES_KEY",
    "Outcome", "RUNS_KEY", "TICK_SECONDS", "class_map_for", "created_text", "describe_run",
    "ensure_state", "execute", "find_saved_run", "load_run", "load_session_runs", "obtain_run",
    "register_run", "run_id_of", "run_stamp",
]
