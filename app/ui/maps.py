"""Citra asli dan peta kelas berdampingan (Langkah 6, butir 4).

UI tidak pernah memuat citra resolusi penuh (plan Bagian 3, aturan 2). Logika murni (`class_map_image`,
`find_original`, `load_original`, `original_preview`) ada di `services/maps.py` (dipakai juga oleh API
FastAPI); modul ini menambah cache Streamlit dan menggambar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from landcover.contract import RunResult
from landcover.labeling import ClassMap
from services import maps as _svc
from services.compare import MOCK_NOTICE
from services.maps import (
    ORIGINAL_MAX_PIXELS,
    ORIGINAL_MAX_SIDE,
    VIEW_CENTROID,
    VIEW_CLASS,
    VIEW_LABELS,
    class_map_image,
    find_original,
)
from ui.badges import MOCK_BADGE

KEY_VIEW = "map_view"


def load_original(path: Path | str, max_side: int = ORIGINAL_MAX_SIDE) -> np.ndarray:
    """`services.maps.load_original` dengan batas piksel modul ini (`ORIGINAL_MAX_PIXELS`)."""
    return _svc.load_original(path, max_side, max_pixels=ORIGINAL_MAX_PIXELS)


@st.cache_data(show_spinner=False, max_entries=16)
def _load_original_cached(path: str, stamp: int) -> np.ndarray:   # noqa: ARG001
    return load_original(path)


def original_preview(run: RunResult, data_dir: Path | str) -> tuple[np.ndarray | None, str | None]:
    """(citra, None) atau (None, alasan tidak tersedia); berkas dibaca lewat cache Streamlit."""
    return _svc.original_preview(
        run, data_dir, loader=lambda path: _load_original_cached(str(path), path.stat().st_mtime_ns)
    )


def render_maps(
    run: RunResult,
    class_map: ClassMap,
    data_dir: Path | str,
) -> None:
    """Dua kolom: citra asli (pratinjau) | peta kelas (pratinjau), dengan toggle di atasnya."""
    view = st.radio(
        "Warna peta", list(VIEW_LABELS), key=KEY_VIEW, format_func=VIEW_LABELS.get, horizontal=True,
        help="Warna centroid = warna rata-rata tiap cluster. Warna kelas dipakai setelah cluster diberi nama.",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Citra asli (pratinjau)")
        original, why_not = original_preview(run, data_dir)
        if original is not None:
            st.image(original, width="stretch")
        else:
            st.info(why_not)
    with right:
        title = "Peta kelas (pratinjau)"
        # Lencana pendek di judul (judul panjang membungkus dan membuat kedua gambar tidak
        # sejajar); kalimat penuh di caption di bawah peta.
        st.subheader(f"{title} {MOCK_BADGE}" if run.mock else title)
        try:
            image = class_map_image(run, class_map, view)
        except ValueError as exc:      # hasil janggal (label atau centroid di luar rentang): jangan mogok
            st.warning(f"Peta warna kelas tidak bisa dibuat ({exc}); ditampilkan warna centroid.")
            image = run.map_preview
        st.image(image, width="stretch")
        if run.mock:
            st.caption(MOCK_NOTICE)


__all__ = [
    "KEY_VIEW", "VIEW_CENTROID", "VIEW_CLASS", "VIEW_LABELS", "class_map_image",
    "find_original", "load_original", "original_preview", "render_maps",
]
