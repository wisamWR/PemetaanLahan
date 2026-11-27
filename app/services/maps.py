"""Citra asli dan peta kelas pratinjau (logika murni, tanpa Streamlit).

Diekstrak dari `ui/maps.py`. UI tidak pernah memuat citra resolusi penuh (plan Bagian 3, aturan 2):
peta kelas dari pratinjau di folder run, citra asli dari `preview.jpg` / PNG kecil yang dikecilkan.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from landcover.contract import RunResult
from landcover.datasets import KIND_PNG, resolve_dataset
from landcover.labeling import ClassMap
from landcover.render import class_palette, recolor

VIEW_CENTROID = "centroid"
VIEW_CLASS = "class"
VIEW_LABELS = {VIEW_CENTROID: "Warna centroid", VIEW_CLASS: "Warna kelas"}

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


def load_original(
    path: Path | str, max_side: int = ORIGINAL_MAX_SIDE, max_pixels: int | None = None
) -> np.ndarray:
    """Baca citra dan kecilkan ke sisi terpanjang <= `max_side`. ValueError bila terlalu besar
    (lebih dari `max_pixels`; bawaan `ORIGINAL_MAX_PIXELS`)."""
    limit = ORIGINAL_MAX_PIXELS if max_pixels is None else max_pixels
    with Image.open(path) as im:
        w, h = im.size
        if w * h > limit:
            raise ValueError(
                f"{Path(path).name} berukuran {w}x{h} (terlalu besar untuk pratinjau); "
                "sediakan preview.jpg di folder dataset"
            )
        rgb = im.convert("RGB")
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


@lru_cache(maxsize=16)
def _load_original_cached(path: str, stamp: int) -> np.ndarray:   # noqa: ARG001 (stamp = kunci cache)
    return load_original(path)


def load_original_cached(path: Path) -> np.ndarray:
    """`load_original` dengan cache berkunci (path, waktu ubah); untuk API."""
    return _load_original_cached(str(path), Path(path).stat().st_mtime_ns)


def original_preview(
    run: RunResult,
    data_dir: Path | str,
    loader: Callable[[Path], np.ndarray] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """(citra, None) atau (None, alasan tidak tersedia). `loader` mengganti pembaca berkas
    (Streamlit memberi versi ber-cache `st.cache_data`)."""
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
        return (loader or load_original_cached)(path), None
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return None, f"Citra asli tidak bisa ditampilkan: {exc}"


JPEG_QUALITY = 88
THUMB_MIN_SIDE, THUMB_MAX_SIDE = 32, 2048


def thumbnail(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Kecilkan citra `(h, w, 3)` uint8 ke sisi terpanjang <= `max_side` (tidak pernah memperbesar)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"citra harus (h, w, 3) uint8 tidak kosong, ditemukan {arr.shape} bertipe {arr.dtype}")
    h, w = arr.shape[:2]
    scale = min(1.0, int(max_side) / max(w, h))
    if scale >= 1.0:
        return arr
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(arr).resize(size, Image.BILINEAR), dtype=np.uint8)


def jpeg_bytes(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG dari citra RGB `(h, w, 3)` uint8, untuk gambar yang hanya DITAMPILKAN (kartu citra dan
    "citra asli"); peta kelas dan ekspor memakai PNG tanpa kehilangan (`landcover.export.png_bytes`)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"citra harus (h, w, 3) uint8 tidak kosong, ditemukan {arr.shape} bertipe {arr.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


__all__ = [
    "JPEG_QUALITY", "ORIGINAL_MAX_PIXELS", "ORIGINAL_MAX_SIDE", "THUMB_MAX_SIDE", "THUMB_MIN_SIDE",
    "VIEW_CENTROID", "VIEW_CLASS", "VIEW_LABELS", "class_map_image", "find_original", "jpeg_bytes",
    "load_original", "load_original_cached", "original_preview", "thumbnail",
]
