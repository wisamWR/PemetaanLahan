"""Entry point aplikasi Streamlit: pemetaan tutupan lahan (K-Means Sentinel-2).

Langkah 6: layout, state, tombol Jalankan/Bandingkan, kartu perbandingan, dan peta.
Langkah 7: grafik waktu per tahap. Langkah 8: panel pelabelan dan tabel luas.
Slot untuk verifikasi (9) sudah disediakan.

Skrip ini sengaja tipis: logika ada di `ui/` (murni dan teruji), tampilan di sini.
`config.BACKEND`, `config.DATA_DIR`, `config.RUNS_DIR` dibaca lewat atribut modul pada
setiap eksekusi (bukan `from config import ...`), supaya tes bisa menimpanya.
"""

from __future__ import annotations

import streamlit as st

import config
from landcover.contract import RunResult, validate
from landcover.datasets import list_datasets
from landcover.labeling import ClassMap
from launchers import make_launcher
from launchers.base import RunParams
from ui import runs as run_state
from ui.badges import backend_badge
from ui.charts import render_stage_slot
from ui.compare import MOCK_NOTICE, comparable_runs, render_cards, render_timer
from ui.labeling_panel import render_labeling_slot
from ui.maps import KEY_VIEW, VIEW_CLASS, render_maps
from ui.sidebar import ACTION_COMPARE, Submission, build_pair, build_params, render_sidebar

TITLE = "Pemetaan Tutupan Lahan"
SUBTITLE = "K-Means piksel RGB Sentinel-2: CPU, OpenMP, dan CUDA"
KEY_PICK = "active_run_pick"
PROGRESS_LINES = 6        # baris progres terakhir yang ditampilkan saat run berjalan


# --- menjalankan ---------------------------------------------------------- #

def _run_steps(submission: Submission) -> list[tuple[str, RunParams]]:
    v = submission.values
    if submission.action == ACTION_COMPARE:
        cpu, opt = build_pair(v)
        return [(run_state.MODE_LABELS["cpu"], cpu), (run_state.MODE_LABELS["opt"], opt)]
    params = build_params(v)
    return [(run_state.MODE_LABELS[params.mode], params)]


def perform(submission: Submission, live) -> None:
    """Jalankan run yang diminta (satu, atau CPU lalu Optimalisasi) dan catat hasilnya di state."""
    try:
        steps = _run_steps(submission)
        launcher = make_launcher(config.BACKEND, config.DATA_DIR)
    except (ValueError, NotImplementedError) as exc:
        live.error(str(exc))
        return

    for label, params in steps:
        timer, log = live.empty(), live.empty()
        lines: list[str] = []

        def on_progress(line: str, _log=log, _lines=lines) -> None:
            _lines.append(line)
            _log.code("\n".join(_lines[-PROGRESS_LINES:]), language=None)

        try:
            outcome = run_state.obtain_run(
                launcher, config.RUNS_DIR, config.BACKEND, params,
                force=submission.values.force,
                on_tick=lambda s, _t=timer, _l=label: render_timer(_t, _l, s),
                on_progress=on_progress,
            )
        except Exception as exc:     # batas UI: galat launcher tampil sebagai pesan, bukan crash
            timer.empty()
            live.error(f"Run '{label}' gagal: {type(exc).__name__}: {exc}")
            return
        timer.empty()
        log.empty()
        run_state.register_run(st.session_state, outcome.run_id, outcome.path, outcome.note)


# --- hasil ---------------------------------------------------------------- #

def _on_pick() -> None:
    st.session_state[run_state.ACTIVE_KEY] = st.session_state[KEY_PICK]


def render_slots(
    cpu: RunResult | None = None,
    opt: RunResult | None = None,
    active: RunResult | None = None,
    class_map: ClassMap | None = None,
) -> bool:
    """Slot di bawah peta: grafik waktu (Langkah 7, `cpu`/`opt` = pasangan sebanding), panel
    pelabelan (Langkah 8, `active` + `class_map` run aktif) lalu tempat Langkah 9. Tanpa run,
    grafik dan panel tampil sebagai kotak kosong. Mengembalikan True bila panel mengubah
    nama kelas pada eksekusi ini (`class_map` diubah di tempat)."""
    render_stage_slot(cpu, opt)
    changed = render_labeling_slot(active, class_map)
    with st.container(border=True):
        st.caption("Status verifikasi antarmode (Langkah 9): belum dibuat")
    return changed


def render_results(banner) -> None:
    state = st.session_state
    if not state[run_state.RUNS_KEY]:
        st.info("Belum ada hasil. Pilih dataset di sidebar, lalu tekan Jalankan atau Bandingkan.")
        render_slots()
        return

    loaded, failed = run_state.load_session_runs(state)
    for rid, why in failed.items():
        st.warning(f"Hasil {rid} tidak terbaca: {why}")
    if not loaded:
        st.error("Semua hasil di sesi ini tidak terbaca; jalankan ulang.")
        return

    active_id = state.get(run_state.ACTIVE_KEY)
    if active_id not in loaded:
        active_id = list(loaded)[-1]
        state[run_state.ACTIVE_KEY] = active_id
    notes = state[run_state.NOTES_KEY]

    # Pemilih hasil: nilai disamakan dulu dengan state, baru widget dibuat (aturan Streamlit:
    # state widget tidak boleh diubah sesudah widget dibuat di eksekusi yang sama).
    state[KEY_PICK] = active_id
    st.selectbox(
        "Hasil yang ditampilkan", list(reversed(loaded)), key=KEY_PICK, on_change=_on_pick,
        format_func=lambda rid: run_state.describe_run(loaded[rid], notes.get(rid)),
    )
    active = loaded[active_id]
    if active.mock:
        banner.warning(MOCK_NOTICE)
    problems = validate(active)
    if problems:
        st.warning("Hasil ini terbaca tetapi janggal:\n" + "\n".join(f"- {p}" for p in problems))

    cpu, opt = comparable_runs(loaded, active_id)
    render_cards(cpu, opt, notes, mock=active.mock)

    class_map = run_state.class_map_for(state, active_id, active.k)
    # Panel pelabelan (di bawah) menentukan nama kelas pada eksekusi ini, jadi peta digambar
    # SESUDAH panel ke dalam wadah yang dibuat di sini: peta langsung memakai nama terbaru.
    maps_box = st.container()
    if render_slots(cpu, opt, active, class_map):
        state[KEY_VIEW] = VIEW_CLASS      # tampilkan warna kelas begitu pengguna memberi nama
    with maps_box:
        render_maps(active, class_map, config.DATA_DIR)


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    run_state.ensure_state(st.session_state)
    st.title(TITLE)
    st.caption(SUBTITLE)
    st.markdown(backend_badge(config.BACKEND))
    banner = st.container()      # diisi sesudah run selesai (peringatan DATA TIRUAN)
    live = st.container()        # timer dan progres selama run berjalan

    scan = list_datasets(config.DATA_DIR, config.BACKEND)
    submission = render_sidebar(scan, config.DATA_DIR, st.session_state)
    if submission is not None:
        perform(submission, live)
    render_results(banner)


main()
