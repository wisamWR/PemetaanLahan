"""Citra asli dan peta kelas berdampingan (Langkah 6, butir 4).

UI tidak pernah memuat citra resolusi penuh (plan Bagian 3, aturan 2): peta kelas dari
pratinjau di folder run, citra asli dari `preview.jpg` / PNG kecil yang dikecilkan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from landcover.contract import RunResult
from landcover.datasets import KIND_PNG, resolve_dataset
from landcover.labeling import ClassMap
from landcover.render import class_palette, recolor
from ui.badges import MOCK_BADGE
from ui.compare import MOCK_NOTICE

VIEW_CENTROID = "centroid"
VIEW_CLASS = "class"
VIEW_LABELS = {VIEW_CENTROID: "Warna centroid", VIEW_CLASS: "Warna kelas"}
KEY_VIEW = "map_view"

ORIGINAL_MAX_SIDE = 1024
ORIGINAL_MAX_PIXELS = 16_000_000     # di atas ini: tolak, jangan dekode (pakai preview.jpg)


def class_map_image(run: RunResult, class_map: ClassMap, view: str) -> np.ndarray:
    """Pratinjau peta `(h, w, 3)` uint8. `warna centroid` = `map_preview.png` apa adanya
    (sama dengan `recolor` dengan semua cluster tanpa nama; diuji pada Langkah 5)."""
    if view == VIEW_CENTROID:
        return run.map_preview
    if view != VIEW_CLASS:
        raise ValueError(f"view harus {VIEW_CENTROID!r} atau {VIEW_CLASS!r}, ditemukan {view!r}")
    return recolor(run.labels_preview, class_palette(class_map, run.centroids))


def find_original(data_dir: Path | str, dataset: str) -> Path | None:
    """Berkas citra untuk "citra asli": `preview.jpg` di folder dataset, atau PNG lepas."""
    preview = Path(data_dir) / dataset / "preview.jpg"
    if preview.is_file():
        return preview
    ds = resolve_dataset(data_dir, dataset)
    if ds is not None and ds.kind == KIND_PNG:
        return ds.path
    return None


def load_original(path: Path | str, max_side: int = ORIGINAL_MAX_SIDE) -> np.ndarray:
    """Baca citra dan kecilkan ke sisi terpanjang <= `max_side`. ValueError bila terlalu besar."""
    with Image.open(path) as im:
        w, h = im.size
        if w * h > ORIGINAL_MAX_PIXELS:
            raise ValueError(
                f"{Path(path).name} berukuran {w}x{h} (terlalu besar untuk pratinjau); "
                "sediakan preview.jpg di folder dataset"
            )
        rgb = im.convert("RGB")
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


@st.cache_data(show_spinner=False, max_entries=16)
def _load_original_cached(path: str, stamp: int) -> np.ndarray:   # noqa: ARG001
    return load_original(path)


def original_preview(run: RunResult, data_dir: Path | str) -> tuple[np.ndarray | None, str | None]:
    """(citra, None) atau (None, alasan tidak tersedia)."""
    if run.meta.get("source_image") == "synthetic":
        # Dataset sintetis (MockLauncher): citranya hanya bergantung pada nama dan ukuran
        # pratinjau, jadi dibuat ulang persis di sini.
        from launchers.mock import synthetic_image

        return synthetic_image(
            str(run.meta["dataset"]), int(run.meta["preview_width"]), int(run.meta["preview_height"])
        ), None
    path = find_original(data_dir, str(run.meta.get("dataset", "")))
    if path is None:
        return None, "Citra asli tidak ditemukan di data/ (butuh preview.jpg atau PNG kecil)."
    try:
        return _load_original_cached(str(path), path.stat().st_mtime_ns), None
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return None, f"Citra asli tidak bisa ditampilkan: {exc}"


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
