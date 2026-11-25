"""Pelabelan cluster dan tabel luas (plan Langkah 4).

Modul murni pandas/NumPy, tanpa Streamlit. Semua angka berasal dari `counts`
resolusi penuh (jumlah piksel integer per cluster, dari tahap update core),
bukan dari pratinjau, jadi persen dan luas eksak:

  * persen = piksel * 100 / jumlah piksel total; dihitung dari integer dan TIDAK
    dibulatkan di sini (pembulatan hanya saat tampil atau ekspor);
  * luas_ha = piksel * pixel_size_m**2 / 10 000 (10 m: satu piksel = 0,01 ha);
  * cluster berjumlah 0 sah (kontrak U2: K lebih besar dari jumlah warna unik):
    tampil 0 piksel, 0 %, 0 ha. Tidak ada pembagian dengan jumlah piksel kelas;
    satu-satunya pembagi adalah total piksel citra, dan total 0 dijaga (hasil 0 %).

Masukan yang melanggar kontrak (counts negatif atau bukan integer, bentuk centroid
salah, ukuran piksel <= 0, id cluster di luar 0..K-1) melempar `ValueError`;
kewajaran isi folder run diperiksa `contract.validate()`, bukan di sini.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import pandas as pd

CLASS_PRESETS = ["Vegetasi", "Air", "Lahan terbangun", "Lahan terbuka", "Awan/bayangan", "Lainnya"]
UNNAMED = "Belum diberi nama"
TOTAL_LABEL = "Total"            # nama baris jumlah di `area_table`; tidak boleh jadi nama kelas
NODATA_LABEL = "Nodata (dikecualikan)"   # nama baris piksel nodata (0,0,0) di `area_table` (terpisah dari Total)

ClassMap = dict[int, str]        # id cluster -> nama kelas

_M2_PER_HA = 10_000.0


# --------------------------------------------------------------------------- #
# Validasi masukan
# --------------------------------------------------------------------------- #

def _check_counts(counts) -> np.ndarray:
    arr = np.asarray(counts)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"counts harus array 1 dimensi tidak kosong, ditemukan bentuk {arr.shape}")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"counts harus bertipe integer, ditemukan {arr.dtype}")
    if arr.min() < 0:
        raise ValueError(f"counts tidak boleh negatif, ditemukan min {arr.min()}")
    return arr.astype(np.int64)


def _check_centroids(centroids, k: int) -> np.ndarray:
    arr = np.asarray(centroids)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"centroids harus berbentuk (K, 3), ditemukan {arr.shape}")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"centroids harus bertipe integer, ditemukan {arr.dtype}")
    if len(arr) != k:
        raise ValueError(f"jumlah centroid ({len(arr)}) tidak sama dengan jumlah counts ({k})")
    return arr.astype(np.int64)


def _check_pixel_size(pixel_size_m) -> float:
    if isinstance(pixel_size_m, bool) or not isinstance(pixel_size_m, (int, float, np.integer, np.floating)):
        raise ValueError(f"pixel_size_m harus angka, ditemukan {type(pixel_size_m).__name__}")
    size = float(pixel_size_m)
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"pixel_size_m harus > 0 dan terhingga, ditemukan {pixel_size_m!r}")
    return size


def normalize_class_name(name: str) -> str:
    """Rapikan nama kelas: buang spasi tepi; kosong berarti `UNNAMED`.

    Sel kosong dari `st.data_editor` (`None`, `NaN`, `pd.NA`) juga berarti `UNNAMED`.
    Nama `Total` ditolak (`ValueError`, tanpa membedakan huruf besar/kecil) karena
    bentrok dengan baris jumlah di `area_table`. Dipanggil juga oleh UI untuk
    isian "Nama lain" yang bebas.
    """
    if name is None or name is pd.NA or (isinstance(name, float) and math.isnan(name)):
        return UNNAMED
    if not isinstance(name, str):
        raise ValueError(f"nama kelas harus teks, ditemukan {type(name).__name__}")
    cleaned = name.strip()
    if not cleaned:
        return UNNAMED
    if cleaned.casefold() == TOTAL_LABEL.casefold():
        raise ValueError(f'nama kelas "{TOTAL_LABEL}" dicadangkan untuk baris jumlah')
    if cleaned.casefold() == NODATA_LABEL.casefold():
        raise ValueError(f'nama kelas "{NODATA_LABEL}" dicadangkan untuk piksel nodata')
    return cleaned


def _normalized_class_map(class_map: ClassMap, k: int) -> dict[int, str]:
    """Salinan peta kelas dengan kunci int murni dan nama dirapikan.

    Cluster yang tidak ada di `class_map` dianggap `UNNAMED`; kunci di luar 0..K-1
    ditolak supaya state basi (peta kelas dari run lain) tidak lolos diam-diam.
    """
    result = {i: UNNAMED for i in range(k)}
    for key, name in class_map.items():
        try:
            idx = operator.index(key)
        except TypeError:
            raise ValueError(f"id cluster harus integer, ditemukan {key!r}") from None
        if not 0 <= idx < k:
            raise ValueError(f"id cluster {idx} di luar rentang 0..{k - 1}")
        result[idx] = normalize_class_name(name)
    return result


# --------------------------------------------------------------------------- #
# Fungsi publik
# --------------------------------------------------------------------------- #

def default_class_map(k: int) -> ClassMap:
    """Peta kelas awal: semua cluster 0..k-1 belum diberi nama."""
    try:
        k = operator.index(k)
    except TypeError:
        raise ValueError(f"k harus integer, ditemukan {k!r}") from None
    if k < 1:
        raise ValueError(f"k harus >= 1, ditemukan {k}")
    return {i: UNNAMED for i in range(k)}


def _percent(pixels: np.ndarray, total: int) -> np.ndarray:
    """Persen dari jumlah piksel integer; total 0 menghasilkan 0 (bukan NaN)."""
    if total == 0:
        return np.zeros(len(pixels), dtype=np.float64)
    return pixels.astype(np.float64) * 100.0 / float(total)


def _area_ha(pixels: np.ndarray, pixel_size_m: float) -> np.ndarray:
    return pixels.astype(np.float64) * (pixel_size_m * pixel_size_m) / _M2_PER_HA


def _valid_counts(pixels: np.ndarray, nodata_pixels: int, nodata_cluster: int) -> tuple[np.ndarray, int, int]:
    """Kurangi piksel nodata (0,0,0) dari cluster-nya setelah validasi konsistensi (U7)."""
    if isinstance(nodata_pixels, bool) or not isinstance(nodata_pixels, (int, np.integer)):
        raise ValueError(f"nodata_pixels harus integer, ditemukan {nodata_pixels!r}")
    if isinstance(nodata_cluster, bool) or not isinstance(nodata_cluster, (int, np.integer)):
        raise ValueError(f"nodata_cluster harus integer, ditemukan {nodata_cluster!r}")
    nd_px = int(nodata_pixels)
    nd_cl = int(nodata_cluster)
    if nd_px < 0:
        raise ValueError(f"nodata_pixels tidak boleh negatif, ditemukan {nd_px}")
    if nd_px == 0:
        if nd_cl != -1:
            raise ValueError(f"nodata_pixels = 0 tetapi nodata_cluster = {nd_cl} (harus -1)")
        return pixels, 0, -1
    if not 0 <= nd_cl < len(pixels):
        raise ValueError(f"nodata_cluster {nd_cl} di luar rentang 0..{len(pixels) - 1}")
    if nd_px > int(pixels[nd_cl]):
        raise ValueError(f"nodata_pixels ({nd_px}) melebihi piksel cluster {nd_cl} ({int(pixels[nd_cl])})")
    valid = pixels.copy()
    valid[nd_cl] -= nd_px
    return valid, nd_px, nd_cl


def cluster_table(
    counts,
    centroids,
    pixel_size_m: float = 10.0,
    *,
    nodata_pixels: int = 0,
    nodata_cluster: int = -1,
) -> pd.DataFrame:
    """Satu baris per cluster, urut id: cluster, r, g, b, piksel, persen, luas_ha.

    Bila `nodata_pixels > 0`, nodata dikurangi dari `nodata_cluster` dan persen dihitung
    terhadap piksel valid.
    """
    pixels = _check_counts(counts)
    rgb = _check_centroids(centroids, len(pixels))
    size = _check_pixel_size(pixel_size_m)
    valid, _, _ = _valid_counts(pixels, nodata_pixels, nodata_cluster)
    total = int(valid.sum())
    return pd.DataFrame(
        {
            "cluster": np.arange(len(pixels), dtype=np.int64),
            "r": rgb[:, 0],
            "g": rgb[:, 1],
            "b": rgb[:, 2],
            "piksel": valid,
            "persen": _percent(valid, total),
            "luas_ha": _area_ha(valid, size),
        }
    )


def area_table(
    counts,
    class_map: ClassMap,
    pixel_size_m: float = 10.0,
    *,
    nodata_pixels: int = 0,
    nodata_cluster: int = -1,
) -> pd.DataFrame:
    """Tabel luas per kelas: kelas, cluster, piksel, persen, luas_ha, plus baris `Total`.

    Cluster bernama kelas sama digabung. Baris kelas diurut menurut id cluster
    terkecil di tiap kelas; kolom `cluster` berisi id yang digabung sebagai teks
    ("0, 3"), supaya tampil dan diekspor ke CSV apa adanya. Baris `Total` selalu
    menutup kelas valid (persen 100 bila ada piksel valid). Bila `nodata_pixels > 0`,
    nodata dikurangi dari cluster-nya, persen dihitung dari piksel valid, dan baris
    `NODATA_LABEL` ditambahkan terpisah sesudah `Total` (tidak masuk Total). Cluster
    yang seluruhnya nodata tidak dimasukkan ke baris kelas valid.
    """
    pixels = _check_counts(counts)
    size = _check_pixel_size(pixel_size_m)
    names = _normalized_class_map(class_map, len(pixels))
    valid, nd_px, nd_cl = _valid_counts(pixels, nodata_pixels, nodata_cluster)
    pure_nodata = nd_cl if (nd_px > 0 and int(valid[nd_cl]) == 0) else None

    groups: dict[str, list[int]] = {}
    for idx in range(len(pixels)):                    # urutan id => urutan baris
        if idx == pure_nodata:
            continue
        groups.setdefault(names[idx], []).append(idx)

    total_valid = int(valid.sum())
    if groups:
        class_pixels = np.array([valid[ids].sum() for ids in groups.values()], dtype=np.int64)
        table = pd.DataFrame(
            {
                "kelas": list(groups),
                "cluster": [", ".join(str(i) for i in ids) for ids in groups.values()],
                "piksel": class_pixels,
                "persen": _percent(class_pixels, total_valid),
                "luas_ha": _area_ha(class_pixels, size),
            }
        )
    else:
        table = pd.DataFrame(
            {
                "kelas": pd.Series(dtype="object"),
                "cluster": pd.Series(dtype="object"),
                "piksel": pd.Series(dtype=np.int64),
                "persen": pd.Series(dtype=np.float64),
                "luas_ha": pd.Series(dtype=np.float64),
            }
        )

    parts = [
        table,
        pd.DataFrame(
            {
                "kelas": [TOTAL_LABEL],
                "cluster": [""],
                "piksel": np.array([total_valid], dtype=np.int64),
                "persen": [100.0 if total_valid > 0 else 0.0],
                "luas_ha": _area_ha(np.array([total_valid], dtype=np.int64), size),
            }
        ),
    ]
    if nd_px > 0:
        all_pixels = int(pixels.sum())
        nd_arr = np.array([nd_px], dtype=np.int64)
        parts.append(
            pd.DataFrame(
                {
                    "kelas": [NODATA_LABEL],
                    "cluster": [str(nd_cl)],
                    "piksel": nd_arr,
                    "persen": _percent(nd_arr, all_pixels),
                    "luas_ha": _area_ha(nd_arr, size),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
