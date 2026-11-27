"""Kartu citra, pratinjau, dan unggahan untuk halaman Pemetaan (logika murni, tanpa Streamlit).

Dipakai API (`GET /api/datasets`, `GET /api/datasets/{nama}/preview`, `POST /api/datasets`). Aturan:

  * UI tidak pernah memuat citra resolusi penuh: pratinjau datang dari `preview.jpg` di folder dataset,
    dari PNG kecil yang dikecilkan (`maps.load_original_cached`, ditolak bila > `ORIGINAL_MAX_PIXELS`),
    atau, untuk dataset sintetis MockLauncher, dibuat ulang PERSIS pada ukuran pratinjau yang dipakai
    MockLauncher (supaya kartu citra sama dengan "citra asli" di hasil run).
  * Unggahan memakai `landcover.datasets.save_upload` (tidak menimpa, hanya PNG sungguhan) ditambah dua
    batas: byte (`MAX_UPLOAD_BYTES`) dan piksel (`MAX_UPLOAD_PIXELS`, dibaca dari header tanpa dekode),
    supaya setiap citra unggahan selalu bisa ditampilkan sebagai pratinjau.
  * Nama dataset TIDAK PERNAH membentuk path dari masukan klien: pemanggil mencocokkannya dengan hasil
    `list_datasets` lebih dulu.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from landcover.datasets import (
    KIND_FOLDER, KIND_PNG, KIND_SYNTHETIC, Dataset, list_datasets, save_upload,
)
from launchers.mock import DEFAULT_SIZE, PREVIEW_MAX_SIDE, synthetic_image
from services.maps import (
    ORIGINAL_MAX_PIXELS, THUMB_MAX_SIDE, THUMB_MIN_SIDE, find_original, jpeg_bytes,
    load_original_cached, thumbnail,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PIXELS = ORIGINAL_MAX_PIXELS
CARD_THUMB_SIDE = 480                 # sisi terpanjang pratinjau kartu citra (API: `size`)


class UploadTooLargeError(ValueError):
    """Unggahan melewati batas byte atau piksel."""


class PreviewUnavailableError(Exception):
    """Dataset ada, tetapi tidak punya pratinjau yang aman ditampilkan (pesan = alasannya)."""


def upload_limits() -> dict:
    mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    megapixel = MAX_UPLOAD_PIXELS / 1e6
    return {
        "max_bytes": MAX_UPLOAD_BYTES,
        "max_pixels": MAX_UPLOAD_PIXELS,
        "hint": f"PNG, maksimal {mb} MB dan {megapixel:g} juta piksel. Nama berkas jadi nama citra.",
    }


# --- ukuran dan kartu ---------------------------------------------------------------------- #

def _manifest_size(folder: Path) -> tuple[int, int] | None:
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
        width, height = manifest["width"], manifest["height"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    ok = all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (width, height))
    return (width, height) if ok else None


@lru_cache(maxsize=128)
def _png_size(path: str, stamp: int) -> tuple[int, int] | None:   # noqa: ARG001 (stamp = kunci cache)
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def dataset_size(ds: Dataset) -> tuple[int, int] | None:
    """(lebar, tinggi) piksel dataset, atau None bila tidak terbaca."""
    if ds.kind == KIND_SYNTHETIC:
        return DEFAULT_SIZE
    if ds.path is None:
        return None
    if ds.kind == KIND_FOLDER:
        return _manifest_size(ds.path)
    try:
        return _png_size(str(ds.path), ds.path.stat().st_mtime_ns)
    except OSError:
        return None


def size_text(size: tuple[int, int] | None) -> str:
    return "ukuran tidak diketahui" if size is None else f"{size[0]:,} × {size[1]:,} piksel"


def has_preview(data_dir: Path | str, ds: Dataset) -> bool:
    """Apakah `dataset_preview` akan berhasil (tanpa mendekode citra)."""
    if ds.kind == KIND_SYNTHETIC:
        return True
    if find_original(data_dir, ds.name) is None:
        return False
    if ds.kind == KIND_PNG:
        size = dataset_size(ds)
        return size is not None and size[0] * size[1] <= ORIGINAL_MAX_PIXELS
    return True


def dataset_card(data_dir: Path | str, ds: Dataset) -> dict:
    size = dataset_size(ds)
    return {
        "name": ds.name,
        "label": ds.label,
        "kind": ds.kind,
        "width": None if size is None else size[0],
        "height": None if size is None else size[1],
        "size_text": size_text(size),
        "mock": ds.kind == KIND_SYNTHETIC,
        "has_preview": has_preview(data_dir, ds),
    }


def dataset_cards(data_dir: Path | str, backend: str) -> tuple[list[dict], tuple[str, ...]]:
    """(kartu per dataset, catatan dataset yang dilewati) untuk `GET /api/datasets`."""
    scan = list_datasets(data_dir, backend)
    return [dataset_card(data_dir, ds) for ds in scan.datasets], scan.skipped


# --- pratinjau ------------------------------------------------------------------------------ #

@lru_cache(maxsize=8)
def _synthetic_cached(name: str) -> np.ndarray:
    width, height = DEFAULT_SIZE
    scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
    return synthetic_image(name, max(1, round(width * scale)), max(1, round(height * scale)))


def dataset_preview_array(data_dir: Path | str, ds: Dataset) -> np.ndarray:
    """Citra pratinjau `(h, w, 3)` uint8 (sisi terpanjang <= 1024). `PreviewUnavailableError` bila tidak ada."""
    if ds.kind == KIND_SYNTHETIC:
        return _synthetic_cached(ds.name)
    path = find_original(data_dir, ds.name)
    if path is None:
        raise PreviewUnavailableError("Citra ini belum punya pratinjau (butuh preview.jpg di folder dataset).")
    try:
        return load_original_cached(path)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PreviewUnavailableError(f"Pratinjau tidak bisa dibuat: {exc}") from None


def clamp_size(size: int | None, default: int) -> int:
    if size is None:
        return default
    return max(THUMB_MIN_SIDE, min(THUMB_MAX_SIDE, int(size)))


@lru_cache(maxsize=64)
def _preview_jpeg_cached(kind: str, name: str, path: str, stamp: int, side: int, data_dir: str) -> bytes:   # noqa: ARG001
    ds = Dataset(name, kind, Path(path) if path else None)
    return jpeg_bytes(thumbnail(dataset_preview_array(data_dir, ds), side))


def dataset_preview_jpeg(data_dir: Path | str, ds: Dataset, side: int = CARD_THUMB_SIDE) -> tuple[bytes, str]:
    """(JPEG, penanda versi untuk ETag). Berkas sumber diperiksa waktu ubahnya supaya cache tidak basi."""
    source = None if ds.kind == KIND_SYNTHETIC else find_original(data_dir, ds.name)
    try:
        stamp = 0 if source is None else source.stat().st_mtime_ns
    except OSError:
        stamp = 0
    path = "" if ds.path is None else str(ds.path)
    data = _preview_jpeg_cached(ds.kind, ds.name, path, stamp, int(side), str(data_dir))
    return data, f"{ds.kind}-{stamp}-{side}-{len(data)}"


# --- unggah ----------------------------------------------------------------------------------- #

def store_upload(data_dir: Path | str, filename: str, data: bytes) -> Dataset:
    """Simpan PNG unggahan sebagai dataset baru. `UploadTooLargeError` (batas byte/piksel),
    `DatasetExistsError` (nama dipakai), atau `ValueError` (bukan PNG valid, nama tidak aman)."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"berkas {len(data) / 1e6:.1f} MB melebihi batas {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )
    if not data:
        raise ValueError("berkas kosong")
    try:                                          # header saja: ukuran piksel tanpa mendekode
        with Image.open(io.BytesIO(data)) as im:
            fmt, (width, height) = im.format, im.size
    except Exception:                             # bukan gambar: pesannya dari `save_upload`
        fmt, width, height = None, 0, 0
    if fmt == "PNG" and width * height > MAX_UPLOAD_PIXELS:
        raise UploadTooLargeError(
            f"citra {width:,} × {height:,} piksel melebihi batas {MAX_UPLOAD_PIXELS / 1e6:g} juta piksel "
            "(pakai preview.jpg di folder dataset untuk citra besar)"
        )
    return save_upload(data_dir, filename, data)


__all__ = [
    "CARD_THUMB_SIDE", "MAX_UPLOAD_BYTES", "MAX_UPLOAD_PIXELS", "PreviewUnavailableError",
    "UploadTooLargeError", "clamp_size", "dataset_card", "dataset_cards", "dataset_preview_array",
    "dataset_preview_jpeg", "dataset_size", "has_preview", "size_text", "store_upload", "upload_limits",
]
