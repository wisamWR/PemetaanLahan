"""PNG dan CSV sebagai bytes untuk tombol unduh (plan Langkah 5).

Modul murni NumPy/Pillow/pandas, tanpa Streamlit. Pembulatan angka untuk ekspor hanya
terjadi di sini: `labeling` mengembalikan persen dan luas tanpa pembulatan.

Keputusan format CSV (jangan diubah diam-diam):

  * kolom `persen` dan `luas_ha` ditulis dengan tepat 2 desimal (`CSV_DECIMALS`), titik
    sebagai pemisah desimal, tanpa notasi ilmiah; kolom lain (piksel, r, g, b, ...)
    ditulis apa adanya. Persen yang sudah dibulatkan bisa tidak berjumlah 100,00;
    baris `Total` dihitung dari integer sehingga tetap 100,00;
  * pemisah koma, akhir baris `\\n` (bukan `os.linesep`, supaya sama di Windows dan WSL),
    UTF-8 TANPA BOM (BOM membuat nama kolom pertama menjadi `\\ufeffkelas` di pembaca
    biasa); kolom `cluster` seperti `0, 2` otomatis diberi tanda kutip.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

CSV_DECIMALS = 2
ROUNDED_COLUMNS = ("persen", "luas_ha")


def png_bytes(rgb) -> bytes:
    """PNG (lossless) dari citra RGB `(h, w, 3)` uint8, untuk `st.download_button`."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"rgb harus (h, w, 3) uint8, ditemukan bentuk {arr.shape} bertipe {arr.dtype}"
        )
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"rgb tidak boleh kosong, ditemukan bentuk {arr.shape}")
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _format_fixed(series: pd.Series, column: str, decimals: int) -> pd.Series:
    values = series.astype("float64")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"kolom {column}: berisi NaN atau tak terhingga, tidak bisa diekspor")
    return values.map(lambda v: f"{v:.{decimals}f}")


def table_csv_bytes(df: pd.DataFrame, *, decimals: int = CSV_DECIMALS) -> bytes:
    """CSV (UTF-8, tanpa indeks) dari `area_table` atau `cluster_table`.

    Hanya kolom `persen` dan `luas_ha` yang dibulatkan ke `decimals` desimal (tetap 2
    desimal, mis. `50.00`); `df` yang masuk tidak diubah. Format selengkapnya di
    docstring modul.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df harus pandas.DataFrame, ditemukan {type(df).__name__}")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 10:
        raise ValueError(f"decimals harus integer 0..10, ditemukan {decimals!r}")

    out = df.copy()
    for column in ROUNDED_COLUMNS:
        if column in out.columns:
            out[column] = _format_fixed(out[column], column, decimals)
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8")
