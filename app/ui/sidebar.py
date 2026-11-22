"""Sidebar: pengunggah citra kecil dan form parameter run (plan Langkah 6, butir 1-2).

Pemisahan: `FormValues`, `build_params`, dan `build_pair` murni dan ada di `services/params.py`
(dipakai juga oleh API FastAPI); modul ini mengimpornya, jadi nama-nama lama tetap berlaku.
`render_sidebar` hanya menggambar widget dan mengembalikan `Submission` bila salah satu tombol
ditekan. Semua widget punya `key` tetap (dipakai tes dan menjaga state).

Form (`st.form`) membuat halaman tidak dijalankan ulang tiap slider digeser. Pengunggah
sengaja di luar form: ia perlu menyimpan berkas lalu memperbarui daftar dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, MutableMapping

import streamlit as st

from landcover.datasets import DatasetExistsError, DatasetScan, save_upload
from landcover.reference_kmeans import KMEANS_MAX_K, MAX_SEED
from launchers.base import KMEANS_CHOICES, MODES
from services.params import (
    ITERS_DEFAULT,
    ITERS_MAX,
    K_DEFAULT,
    K_MIN,
    SEED_DEFAULT,
    THREADS_DEFAULT,
    THREADS_MAX,
    THREADS_MIN,
    FormValues,
    build_pair,
    build_params,
)
from ui.runs import KMEANS_LABELS, MODE_LABELS

if TYPE_CHECKING:
    from launchers.cli import CoreInfo

ACTION_RUN = "run"
ACTION_COMPARE = "compare"

# Kunci widget (tetap, supaya tes dan state stabil).
KEY_DATASET = "f_dataset"
KEY_K = "f_k"
KEY_ITERS = "f_iters"
KEY_MODE = "f_mode"
KEY_SEED = "f_seed"
KEY_KMEANS = "f_kmeans"
KEY_THREADS = "f_threads"
KEY_FORCE = "f_force"
KEY_BTN_RUN = "btn_run"
KEY_BTN_COMPARE = "btn_compare"
KEY_UPLOAD = "upl_file"
KEY_UPLOAD_SAVE = "upl_save"
UPLOAD_COUNTER = "_upl_counter"
FLASH = "_flash"


@dataclass(frozen=True)
class Submission:
    action: str                        # ACTION_RUN | ACTION_COMPARE
    values: FormValues


def _upload_section(data_dir: Path, state: MutableMapping) -> None:
    n = state.setdefault(UPLOAD_COUNTER, 0)
    with st.expander("Unggah citra kecil (PNG)"):
        st.caption("Hanya untuk citra kecil (batas bawaan Streamlit 200 MB). Tile penuh: taruh di data/.")
        file = st.file_uploader("Berkas PNG", type=["png"], key=f"{KEY_UPLOAD}_{n}")
        if st.button("Simpan sebagai dataset", key=KEY_UPLOAD_SAVE, disabled=file is None):
            try:
                ds = save_upload(data_dir, file.name, file.getvalue())
            except DatasetExistsError as exc:
                st.error(str(exc))
            except (ValueError, OSError) as exc:
                st.error(f"Unggahan ditolak: {exc}")
            else:
                state[KEY_DATASET] = ds.name           # dipilih otomatis; form dibuat sesudah ini
                state[UPLOAD_COUNTER] = n + 1          # kunci baru = kotak unggah kosong lagi
                state[FLASH] = f"Dataset '{ds.name}' tersimpan di data/uploads/."
                st.rerun()


def render_sidebar(
    scan: DatasetScan,
    data_dir: Path,
    state: MutableMapping,
    engines: "CoreInfo | None" = None,
) -> Submission | None:
    """Gambar sidebar; kembalikan `Submission` bila "Jalankan" atau "Bandingkan" ditekan.

    `engines` (hasil `landcover --info`, backend cli): pilihan K-Means yang tidak terhubung pada
    build itu (OpenMP/CUDA) tidak ditawarkan, dan `engines.notice` ditampilkan di bawah pilihan
    mode. None = semua pilihan (backend mock)."""
    kmeans_choices = engines.kmeans_choices if engines is not None else KMEANS_CHOICES
    if state.get(KEY_KMEANS) not in kmeans_choices:
        state.pop(KEY_KMEANS, None)                    # pilihan basi (mesin hilang): kembali ke "auto"
    with st.sidebar:
        st.header("Parameter")
        _upload_section(Path(data_dir), state)
        flash = state.pop(FLASH, None)
        if flash:
            st.success(flash)
        if scan.skipped:
            with st.expander(f"{len(scan.skipped)} berkas dilewati"):
                for line in scan.skipped:
                    st.caption(line)

        names = list(scan.names)
        if state.get(KEY_DATASET) not in names:
            state.pop(KEY_DATASET, None)               # dataset hilang dari disk: pilih ulang
        if not names:
            st.warning(f"Tidak ada dataset di {data_dir} (folder tile dengan manifest.json, atau PNG).")
        labels = {d.name: d.label for d in scan.datasets}

        with st.form("params"):
            dataset = st.selectbox(
                "Dataset", names, key=KEY_DATASET, format_func=lambda n: labels.get(n, n),
                help="Folder data/<nama>/ dengan manifest.json, atau PNG kecil.",
            )
            k = st.slider("Jumlah cluster (K)", K_MIN, KMEANS_MAX_K, K_DEFAULT, key=KEY_K)
            iters = st.number_input(
                "Iterasi", min_value=1, max_value=ITERS_MAX, value=ITERS_DEFAULT, step=1, key=KEY_ITERS,
            )
            mode = st.radio(
                "Mode", MODES, key=KEY_MODE, format_func=MODE_LABELS.get,
                help="Dipakai oleh tombol Jalankan. Bandingkan menjalankan keduanya.",
            )
            if engines is not None and engines.notice:
                st.caption(engines.notice)
            with st.expander("Lanjutan"):
                seed = st.number_input(
                    "Seed inisialisasi", min_value=0, max_value=MAX_SEED, value=SEED_DEFAULT,
                    step=1, key=KEY_SEED,
                )
                kmeans = st.selectbox(
                    "K-Means (mode Optimalisasi)", kmeans_choices, key=KEY_KMEANS,
                    format_func=KMEANS_LABELS.get,
                )
                threads = st.slider(
                    "Thread OpenMP", THREADS_MIN, THREADS_MAX, THREADS_DEFAULT, key=KEY_THREADS,
                )
                force = st.checkbox(
                    "Paksa jalankan ulang", key=KEY_FORCE,
                    help="Abaikan hasil CPU saja yang tersimpan di runs/ dan hitung ulang.",
                )
            disabled = not names
            run = st.form_submit_button("Jalankan", key=KEY_BTN_RUN, type="primary",
                                        disabled=disabled, width="stretch")
            compare = st.form_submit_button("Bandingkan (CPU saja lalu Optimalisasi)",
                                            key=KEY_BTN_COMPARE, disabled=disabled, width="stretch")

    if not (run or compare) or dataset is None:
        return None
    values = FormValues(
        dataset=dataset, k=int(k), iters=int(iters), mode=mode, kmeans=kmeans,
        threads=int(threads), seed=int(seed), force=bool(force),
    )
    return Submission(ACTION_COMPARE if compare else ACTION_RUN, values)


__all__ = [
    "ACTION_COMPARE", "ACTION_RUN", "FormValues", "Submission", "build_pair", "build_params",
    "render_sidebar",
]
