"""Logika pelabelan cluster dan tabel luas (murni, tanpa Streamlit).

Diekstrak dari `ui/labeling_panel.py` (aturan lengkap ada di docstring modul itu): editor Streamlit
dan API memakai fungsi yang sama untuk nama kelas, ringkasan luas, nama berkas unduhan, CSV, PNG peta
kelas, dan PNG resolusi penuh.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from landcover.contract import RunResult
from landcover.export import png_bytes, table_csv_bytes
from landcover.labeling import (
    CLASS_PRESETS, TOTAL_LABEL, UNNAMED, ClassMap, _normalized_class_map, area_table,
    cluster_table, default_class_map, normalize_class_name,
)
from landcover.render import class_palette, export_full_res_png, recolor
from services.runs import run_id_of

KEY_EDITOR_PREFIX = "label_editor__"
LABELS_FILE = "labels.u8"                 # keluaran `landcover --write-labels` (Langkah 11a)
DEFAULT_PIXEL_SIZE_M = 10.0
MOCK_FILE_PREFIX = "TIRUAN_"

COL_COLOR = "Warna"
COL_CLUSTER = "Cluster"
COL_PIXELS = "Piksel"
COL_PERCENT = "Persen (%)"
COL_CLASS = "Kelas"
COL_OTHER = "Nama lain"
EDITOR_COLUMNS = (COL_COLOR, COL_CLUSTER, COL_PIXELS, COL_PERCENT, COL_CLASS, COL_OTHER)
READ_ONLY_COLUMNS = (COL_COLOR, COL_CLUSTER, COL_PIXELS, COL_PERCENT)

SWATCH_SIZE = (40, 20)


def editor_key(run_id: str) -> str:
    return f"{KEY_EDITOR_PREFIX}{run_id}"


def pixel_size_of(run: RunResult) -> tuple[float, str | None]:
    """(ukuran piksel m, catatan). `pixel_size_m` tidak wajib di meta.json (kontrak); bila
    tidak ada atau tidak masuk akal, dipakai 10 m dan alasannya dikembalikan sebagai catatan."""
    raw = run.meta.get("pixel_size_m")
    if raw is None:
        return DEFAULT_PIXEL_SIZE_M, f"meta.json tidak mencatat pixel_size_m; luas dihitung dengan {DEFAULT_PIXEL_SIZE_M:g} m."
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw <= 0:
        return DEFAULT_PIXEL_SIZE_M, f"pixel_size_m di meta.json tidak valid ({raw!r}); luas dihitung dengan {DEFAULT_PIXEL_SIZE_M:g} m."
    return float(raw), None


@lru_cache(maxsize=512)
def swatch_data_url(rgb: tuple[int, int, int]) -> str:
    """PNG kecil satu warna sebagai data URL, untuk `ImageColumn`. Nilai di luar 0..255 dipotong
    (hanya untuk tampilan; `contract.validate()` yang memperingatkan centroid janggal)."""
    color = tuple(min(255, max(0, int(c))) for c in rgb)
    buffer = io.BytesIO()
    Image.new("RGB", SWATCH_SIZE, color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def split_name(name: str) -> tuple[str | None, str]:
    """Nama kelas -> (isi kolom Kelas atau None, isi kolom Nama lain)."""
    name = normalize_class_name(name)
    if name == UNNAMED:
        return None, ""
    if name in CLASS_PRESETS:
        return name, ""
    return None, name


def _blank(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip()
    )


def resolve_name(kelas, nama_lain) -> str:
    """Nama akhir satu cluster dari dua kolom editor (Nama lain mengalahkan Kelas).

    `ValueError` untuk nama yang ditolak `normalize_class_name` (`Total`, tipe bukan teks)."""
    if not _blank(nama_lain):
        return normalize_class_name(nama_lain)
    return normalize_class_name(None if _blank(kelas) else kelas)


def editor_frame(counts, centroids, class_map: ClassMap, pixel_size_m: float = DEFAULT_PIXEL_SIZE_M) -> pd.DataFrame:
    """Isi awal editor: satu baris per cluster, kolom `EDITOR_COLUMNS`, dtype stabil."""
    table = cluster_table(counts, centroids, pixel_size_m)
    names = _normalized_class_map(class_map, len(table))
    split = [split_name(names[i]) for i in range(len(table))]
    return pd.DataFrame(
        {
            COL_COLOR: [swatch_data_url((int(r), int(g), int(b))) for r, g, b in zip(table["r"], table["g"], table["b"])],
            COL_CLUSTER: table["cluster"].to_numpy(dtype=np.int64),
            COL_PIXELS: table["piksel"].to_numpy(dtype=np.int64),
            COL_PERCENT: table["persen"].to_numpy(dtype=np.float64),
            COL_CLASS: pd.Series([kelas for kelas, _ in split], dtype="string"),
            COL_OTHER: pd.Series([other for _, other in split], dtype="string"),
        },
        columns=list(EDITOR_COLUMNS),
    )


def class_map_from_frame(frame: pd.DataFrame, previous: ClassMap, k: int) -> tuple[ClassMap, list[str]]:
    """Peta kelas baru dari tabel editor: (peta, daftar galat).

    Cluster dipetakan lewat kolom `Cluster` (bukan posisi baris). Nama yang ditolak
    mempertahankan nama sebelumnya dan masuk ke daftar galat; fungsi ini tidak memanggil
    `st.*`. `previous` tidak diubah. `ValueError` bila kolom wajib hilang atau id cluster di
    luar 0..k-1 (itu kesalahan program, bukan isian pengguna)."""
    missing = [c for c in (COL_CLUSTER, COL_CLASS, COL_OTHER) if c not in frame.columns]
    if missing:
        raise ValueError(f"kolom editor hilang: {', '.join(missing)}")
    result = _normalized_class_map(previous, k)
    errors: list[str] = []
    for cluster, kelas, other in zip(frame[COL_CLUSTER], frame[COL_CLASS], frame[COL_OTHER]):
        idx = int(cluster)
        if not 0 <= idx < k:
            raise ValueError(f"id cluster {idx} di luar rentang 0..{k - 1}")
        try:
            result[idx] = resolve_name(kelas, other)
        except ValueError as exc:
            errors.append(f"Cluster {idx}: {exc}. Nama sebelumnya ({result[idx]}) dipertahankan.")
    return result, errors


@dataclass(frozen=True)
class Summary:
    total_ha: float
    named_classes: int            # kelas berbeda yang sudah diberi nama
    largest: tuple[str, float] | None      # (nama, persen) kelas bernama terbesar; None bila belum ada
    unnamed_percent: float        # persen piksel pada cluster yang belum diberi nama


def summary_of(area: pd.DataFrame) -> Summary:
    """Ringkasan dari `area_table` (baris `Total` dan `UNNAMED` diperlakukan khusus)."""
    body = area[area["kelas"] != TOTAL_LABEL]
    named = body[body["kelas"] != UNNAMED]
    unnamed = body[body["kelas"] == UNNAMED]
    total_ha = float(area.loc[area["kelas"] == TOTAL_LABEL, "luas_ha"].iloc[0])
    largest = None
    if len(named):
        top = named.loc[named["piksel"].idxmax()]
        largest = (str(top["kelas"]), float(top["persen"]))
    return Summary(
        total_ha=total_ha,
        named_classes=len(named),
        largest=largest,
        unnamed_percent=float(unnamed["persen"].sum()) if len(unnamed) else 0.0,
    )


def format_ha(value: float) -> str:
    return f"{value:,.2f} ha"


def format_percent(value: float) -> str:
    return f"{value:.2f} %"


def export_names(run_id: str, mock: bool) -> dict[str, str]:
    """Nama berkas unduhan. Hasil tiruan diberi awalan `TIRUAN_` (lihat docstring modul)."""
    prefix = MOCK_FILE_PREFIX if mock else ""
    return {
        "map": f"{prefix}peta-kelas_{run_id}.png",
        "area": f"{prefix}luas-per-kelas_{run_id}.csv",
        "clusters": f"{prefix}per-cluster_{run_id}.csv",
        "full": f"{prefix}peta-kelas-penuh_{run_id}.png",
    }


def class_map_png(run: RunResult, class_map: ClassMap) -> bytes:
    """PNG pratinjau peta kelas (warna kelas; cluster tanpa nama memakai warna centroid)."""
    return png_bytes(recolor(run.labels_preview, class_palette(class_map, run.centroids)))


def area_csv(run: RunResult, class_map: ClassMap) -> bytes:
    return table_csv_bytes(area_table(run.counts, class_map, pixel_size_of(run)[0]))


def cluster_csv(run: RunResult) -> bytes:
    return table_csv_bytes(cluster_table(run.counts, run.centroids, pixel_size_of(run)[0]))


def labels_file_of(run: RunResult) -> Path | None:
    path = run.path / LABELS_FILE
    return path if path.is_file() else None


def export_full_map(run: RunResult, class_map: ClassMap, out_path: Path | str | None = None) -> Path:
    """Tulis PNG berpalet resolusi penuh dari `labels.u8` ke folder run (atau `out_path`).

    Memakai `run.width`/`run.height` dari meta (bukan ukuran pratinjau). `FileNotFoundError`
    bila `labels.u8` tidak ada; `ValueError` dari `export_full_res_png` bila ukuran berkas atau
    label tidak cocok (tidak ada yang ditulis). Penulisan atomik."""
    source = labels_file_of(run)
    if source is None:
        raise FileNotFoundError(f"{run.path / LABELS_FILE} tidak ada (dibuat oleh landcover --write-labels)")
    target = Path(out_path) if out_path is not None else run.path / export_names(run_id_of(run), run.mock)["full"]
    export_full_res_png(source, run.width, run.height, class_palette(class_map, run.centroids), target)
    return target


# --- API (Fase 2 UI web): sama dengan editor Streamlit, tetapi tanpa DataFrame di klien ----------- #

NAME_MAX_CHARS = 40                       # sama dengan `max_chars` kolom "Nama lain" di editor Streamlit
CLUSTER_ERROR = re.compile(r"^Cluster (\d+):")

# Kalimat bantuan yang sama dengan editor Streamlit (`ui/labeling_panel.py` menulisnya di tempat; ui tidak
# boleh diimpor dari sini karena mengimpor Streamlit).
HELP_PANEL = (
    "Beri nama tiap cluster: pilih Kelas atau ketik Nama lain. Cluster bernama sama digabung, "
    "dan peta langsung memakai warna kelas."
)
HELP_OTHER = (
    "Nama bebas, mis. Sawah. Bila diisi, mengalahkan kolom Kelas; kosongkan agar pilihan Kelas dipakai. "
    "Huruf besar/kecil dibedakan. Nama 'Total' tidak boleh dipakai."
)
HELP_CLASS = "Cluster ber-Kelas sama digabung di tabel luas dan berwarna sama di peta."


def preview_caption(run: RunResult) -> str:
    height, width = run.labels_preview.shape[:2]
    return (
        f"Pratinjau {width:,} × {height:,} piksel dari citra {run.width:,} × {run.height:,}. "
        "Persen dan luas dihitung dari seluruh piksel (counts), bukan dari pratinjau."
    )


def area_caption(size: float, note: str | None) -> str:
    text = f"Persen dari seluruh piksel (counts); luas dihitung dengan {size:g} m per piksel."
    return f"{text} {note}" if note else text


def export_note(run: RunResult) -> str | None:
    return f"Berkas unduhan hasil tiruan berawalan {MOCK_FILE_PREFIX}." if run.mock else None


EXPORT_KINDS = ("map", "area", "clusters", "full")
EXPORT_LABELS = {
    "map": "Peta kelas (PNG pratinjau)",
    "area": "Tabel luas per kelas (CSV)",
    "clusters": "Tabel per cluster (CSV)",
    "full": "PNG resolusi penuh",
}
EXPORT_MEDIA_TYPES = {"map": "image/png", "area": "text/csv", "clusters": "text/csv", "full": "image/png"}


def hex_color(rgb) -> str:
    """`#rrggbb` dari tiga integer (nilai di luar 0..255 dipotong, hanya untuk tampilan)."""
    r, g, b = (min(255, max(0, int(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def palette_hex(run: RunResult, class_map: ClassMap) -> list[str]:
    """Warna peta per cluster (aturan `class_palette`: tanpa nama = warna centroid sendiri)."""
    return [hex_color(c) for c in class_palette(class_map, run.centroids)]


def centroid_hex(run: RunResult) -> list[str]:
    return [hex_color(c) for c in run.centroids]


def _check_name_length(value, what: str) -> None:
    if isinstance(value, str) and len(value.strip()) > NAME_MAX_CHARS:
        raise ValueError(f"{what} terlalu panjang (maksimal {NAME_MAX_CHARS} karakter)")


def class_map_from_rows(rows: Mapping[int, tuple], k: int) -> tuple[ClassMap, list[str]]:
    """Peta kelas dari isian editor `{id cluster: (Kelas, Nama lain)}`: (peta, daftar galat).

    Jalur yang sama dengan editor Streamlit (`class_map_from_frame`), jadi "Nama lain mengalahkan
    Kelas" dan penolakan nama `Total` tidak punya salinan kedua. API tanpa keadaan: `previous` di sini
    selalu peta kosong, jadi cluster dengan nama yang ditolak kembali ke `UNNAMED` (galatnya tertulis
    di daftar). Cluster yang tidak disebut = tanpa nama. `ValueError` untuk id di luar 0..k-1 atau
    nama lebih dari `NAME_MAX_CHARS` karakter."""
    for idx, (kelas, other) in rows.items():
        if not 0 <= idx < k:
            raise ValueError(f"id cluster {idx} di luar rentang 0..{k - 1}")
        _check_name_length(kelas, f"Kelas cluster {idx}")
        _check_name_length(other, f"Nama lain cluster {idx}")
    frame = pd.DataFrame(
        {
            COL_CLUSTER: list(range(k)),
            COL_CLASS: [rows.get(i, (None, None))[0] for i in range(k)],
            COL_OTHER: [rows.get(i, (None, None))[1] for i in range(k)],
        }
    )
    return class_map_from_frame(frame, default_class_map(k), k)


def class_map_from_names(names: Sequence | None, k: int) -> ClassMap:
    """Peta kelas dari daftar nama AKHIR per cluster (indeks = id cluster), mis. dari URL peta/ekspor.

    `None` = semua tanpa nama. Panjang daftar harus K (daftar basi dari run ber-K lain ditolak, bukan
    dipotong). `ValueError` untuk panjang salah, nama bukan teks, nama `Total`, atau terlalu panjang."""
    if names is None:
        return default_class_map(k)
    if len(names) != k:
        raise ValueError(f"jumlah nama ({len(names)}) tidak sama dengan K ({k})")
    result: ClassMap = {}
    for idx, name in enumerate(names):
        if not isinstance(name, str):
            raise ValueError(f"nama cluster {idx} harus teks, ditemukan {type(name).__name__}")
        _check_name_length(name, f"nama cluster {idx}")
        result[idx] = normalize_class_name(name)
    return _normalized_class_map(result, k)


def names_of(class_map: ClassMap, k: int) -> list[str]:
    """Nama akhir per cluster (indeks = id cluster; tanpa nama = teks `UNNAMED`)."""
    normalized = _normalized_class_map(class_map, k)
    return [normalized[i] for i in range(k)]


def cluster_error_ids(errors: Sequence[str]) -> set[int]:
    """Id cluster yang disebut di daftar galat `class_map_from_frame` ("Cluster N: ...")."""
    found = (CLUSTER_ERROR.match(message) for message in errors)
    return {int(m.group(1)) for m in found if m is not None}


def run_fingerprint(run: RunResult) -> str:
    """Sidik hasil (counts + centroid): sama bila run ditimpa dengan hasil identik. Dipakai klien untuk
    memutuskan apakah nama kelas yang tersimpan di browser masih berlaku untuk run ini."""
    payload = json.dumps([[int(c) for c in run.counts], [[int(v) for v in row] for row in run.centroids]])
    return hashlib.sha1(payload.encode("ascii")).hexdigest()[:16]


def export_items(run: RunResult) -> list[dict]:
    """Daftar ekspor yang ditawarkan UI: jenis, label, nama berkas, ketersediaan, dan alasan bila tidak."""
    names = export_names(run_id_of(run), run.mock)
    has_labels = labels_file_of(run) is not None
    items = []
    for kind in EXPORT_KINDS:
        available, reason = True, None
        if kind == "full" and not has_labels:
            available = False
            reason = (
                f"PNG resolusi penuh tidak tersedia: folder run ini tidak punya {LABELS_FILE} "
                "(dibuat oleh landcover --write-labels)."
            )
        items.append({
            "kind": kind,
            "label": EXPORT_LABELS[kind],
            "filename": names[kind],
            "media_type": EXPORT_MEDIA_TYPES[kind],
            "available": available,
            "reason": reason,
            "hint": f"PNG berpalet {run.width:,} × {run.height:,}; bisa beberapa detik." if kind == "full" and available else None,
        })
    return items


__all__ = [
    "HELP_CLASS", "HELP_OTHER", "HELP_PANEL", "area_caption", "export_note", "preview_caption",
    "CLUSTER_ERROR", "COL_CLASS", "COL_CLUSTER", "COL_COLOR", "COL_OTHER", "COL_PERCENT", "COL_PIXELS",
    "DEFAULT_PIXEL_SIZE_M", "EDITOR_COLUMNS", "EXPORT_KINDS", "EXPORT_LABELS", "EXPORT_MEDIA_TYPES",
    "KEY_EDITOR_PREFIX", "LABELS_FILE", "MOCK_FILE_PREFIX", "NAME_MAX_CHARS", "READ_ONLY_COLUMNS",
    "SWATCH_SIZE", "Summary", "area_csv", "centroid_hex", "class_map_from_frame", "class_map_from_names",
    "class_map_from_rows", "class_map_png", "cluster_csv", "cluster_error_ids", "editor_frame",
    "editor_key", "export_full_map", "export_items", "export_names", "format_ha", "format_percent",
    "hex_color", "labels_file_of", "names_of", "palette_hex", "pixel_size_of", "resolve_name",
    "run_fingerprint", "split_name", "summary_of", "swatch_data_url",
]
