"""Palet kelas, pewarnaan ulang pratinjau, dan ekspor PNG resolusi penuh (plan Langkah 5).

Modul murni NumPy/Pillow, tanpa Streamlit. Bagian yang menghasilkan bytes untuk tombol
unduh (PNG pratinjau, CSV) ada di `landcover.export`.

Aturan warna (`class_palette`):

  * cluster `UNNAMED` memakai warna centroid-nya sendiri;
  * cluster bernama yang ada di `CLASS_COLORS` memakai warna tabel itu;
  * cluster bernama lain (isian bebas, mis. "Sawah") memakai `fallback_color(nama)`,
    warna deterministik dari nama. Warna centroid TIDAK boleh dipakai di sini: cluster
    yang digabung (nama sama) harus berwarna sama, padahal centroid-nya berbeda.

Normalisasi `class_map` memakai fungsi `labeling` yang sama dengan `area_table`
(`_normalized_class_map`, diimpor dari dalam paket), jadi aturan "sel kosong = UNNAMED",
"`Total` ditolak", dan "id di luar 0..K-1 ditolak" tidak punya salinan kedua yang bisa
menyimpang. Cluster yang digabung di tabel luas selalu berwarna sama di peta.

Masukan yang melanggar kontrak melempar `ValueError`. Nilai warna dan centroid di luar
0..255 ditolak (tidak dipotong atau dibungkus `astype(uint8)`).
"""

from __future__ import annotations

import colorsys
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from landcover.labeling import UNNAMED, ClassMap, _normalized_class_map

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Vegetasi": (34, 139, 34),
    "Air": (30, 100, 200),
    "Lahan terbangun": (200, 60, 60),
    "Lahan terbuka": (210, 180, 120),
    "Awan/bayangan": (170, 170, 170),
    "Lainnya": (90, 90, 90),
}

_MAX_PNG_PALETTE = 256            # PNG berpalet 8-bit


# --------------------------------------------------------------------------- #
# Validasi masukan
# --------------------------------------------------------------------------- #

def _is_integer_array(arr: np.ndarray) -> bool:
    return arr.dtype != np.bool_ and np.issubdtype(arr.dtype, np.integer)


def _check_rgb_rows(value, what: str) -> np.ndarray:
    """`value` harus (N, 3) integer 0..255, N >= 1; hasil (N, 3) uint8."""
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
        raise ValueError(f"{what} harus berbentuk (N, 3) dengan N >= 1, ditemukan {arr.shape}")
    if not _is_integer_array(arr):
        raise ValueError(f"{what} harus bertipe integer, ditemukan {arr.dtype}")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi > 255:
        raise ValueError(f"{what} harus berisi 0..255, ditemukan min {lo}, max {hi}")
    return arr.astype(np.uint8)


def _check_color(value, what: str) -> np.ndarray:
    """Satu warna: tiga integer 0..255; hasil (3,) uint8. Bool ditolak (NumPy menganggap True = 1)."""
    if isinstance(value, (list, tuple)) and any(isinstance(c, (bool, np.bool_)) for c in value):
        raise ValueError(f"warna {what} harus tiga integer 0..255, ditemukan {value!r}")
    try:
        arr = np.asarray(value)
    except ValueError:                                  # daftar bersarang tak beraturan
        raise ValueError(f"warna {what} harus tiga integer 0..255, ditemukan {value!r}") from None
    if arr.shape != (3,) or not _is_integer_array(arr) or int(arr.min()) < 0 or int(arr.max()) > 255:
        raise ValueError(f"warna {what} harus tiga integer 0..255, ditemukan {value!r}")
    return arr.astype(np.uint8)


def _check_dimension(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{what} harus integer >= 1, ditemukan {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# Palet
# --------------------------------------------------------------------------- #

def fallback_color(name: str) -> tuple[int, int, int]:
    """Warna deterministik untuk nama kelas yang tidak ada di `CLASS_COLORS`.

    Hue dari SHA-256 nama (stabil antarproses; `hash()` Python diacak per proses),
    saturasi dan kecerahan tetap, jadi selalu berwarna jelas dan tidak keabu-abuan.
    Nama persis sama (huruf besar/kecil dibedakan, seperti penggabungan di `labeling`)
    menghasilkan warna persis sama.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    hue = (int.from_bytes(digest[:4], "big") % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def class_palette(class_map: ClassMap, centroids, colors=CLASS_COLORS) -> np.ndarray:
    """Palet (K, 3) uint8: satu warna per cluster menurut aturan di docstring modul.

    `centroids` (K, 3) integer 0..255 menentukan K dan selalu divalidasi, juga bila
    semua cluster sudah bernama. Cluster yang tidak ada di `class_map` dianggap
    `UNNAMED`. Entri `UNNAMED` di `colors` diabaikan (cluster tanpa nama selalu memakai
    warna centroid-nya). Warna di `colors` divalidasi saat dipakai.
    """
    centroid_rgb = _check_rgb_rows(centroids, "centroids")
    k = len(centroid_rgb)
    names = _normalized_class_map(class_map, k)

    palette = np.empty((k, 3), dtype=np.uint8)
    for idx in range(k):
        name = names[idx]
        if name == UNNAMED:
            palette[idx] = centroid_rgb[idx]
        elif name in colors:
            palette[idx] = _check_color(colors[name], f"colors[{name!r}]")
        else:
            palette[idx] = fallback_color(name)
    return palette


def recolor(labels_preview, palette) -> np.ndarray:
    """Peta warna `(h, w, 3)` uint8 dari label pratinjau: `palette[labels]`.

    `labels_preview` 2-D bertipe integer (dari `RunResult`: uint8 grayscale). Label di
    luar 0..K-1 ditolak dengan pesan yang menyebut nilai minimum dan maksimumnya.
    Hasilnya salinan baru; palet dan label tidak berubah.
    """
    pal = _check_rgb_rows(palette, "palette")
    labels = np.asarray(labels_preview)
    if labels.ndim != 2:
        raise ValueError(f"labels harus 2 dimensi (h, w), ditemukan bentuk {labels.shape}")
    if not _is_integer_array(labels):
        raise ValueError(f"labels harus bertipe integer, ditemukan {labels.dtype}")
    if labels.size:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= len(pal):
            raise ValueError(
                f"label di luar rentang 0..{len(pal) - 1}: min {lo}, max {hi} "
                f"(palet hanya punya {len(pal)} warna)"
            )
    return pal[labels]


# --------------------------------------------------------------------------- #
# Ekspor resolusi penuh
# --------------------------------------------------------------------------- #

def export_full_res_png(labels_u8_path, w: int, h: int, palette, out_path) -> None:
    """Tulis `labels.u8` (H*W uint8, row-major) sebagai PNG berpalet ke `out_path`.

    Hanya butuh label uint8 (120 MB untuk satu tile) tanpa membuat citra RGB 362 MB:
    indeks piksel = id cluster, warna di tabel palet PNG. Sebelum membaca, ukuran file
    harus persis `w * h` byte dan label terbesar harus < jumlah warna palet; bila tidak,
    `ValueError` dan tidak ada file yang ditulis. Penulisan atomik (`.tmp` lalu
    `os.replace`), folder tujuan dibuat bila belum ada.

    `w` = lebar, `h` = tinggi (kolom dan baris); file disusun baris demi baris.
    Membaca PNG sebesar tile penuh dengan Pillow memunculkan `DecompressionBombWarning`
    (> 89 juta piksel); itu wajar dan tidak berarti file rusak.
    """
    w = _check_dimension(w, "w")
    h = _check_dimension(h, "h")
    pal = _check_rgb_rows(palette, "palette")
    if len(pal) > _MAX_PNG_PALETTE:
        raise ValueError(f"palette punya {len(pal)} warna; PNG berpalet maksimal {_MAX_PNG_PALETTE}")

    src = Path(labels_u8_path)
    size = src.stat().st_size                       # FileNotFoundError bila tidak ada
    if size != w * h:
        raise ValueError(
            f"{src.name}: ukuran {size} byte, seharusnya w * h = {w} * {h} = {w * h} byte"
        )
    labels = np.fromfile(src, dtype=np.uint8)
    top = int(labels.max())
    if top >= len(pal):
        raise ValueError(
            f"{src.name}: label terbesar {top} di luar palet 0..{len(pal) - 1}"
        )

    image = Image.frombuffer("P", (w, h), labels, "raw", "P", 0, 1)
    image.putpalette(pal.reshape(-1).tolist())

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        image.save(tmp, format="PNG")               # format eksplisit: akhiran .tmp
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
