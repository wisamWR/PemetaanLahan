"""Panel pelabelan cluster dan tabel luas (plan Langkah 8).

Satu baris editor per cluster: contoh warna centroid, id, piksel, persen, kolom **Kelas**
(pilihan `CLASS_PRESETS`) dan kolom **Nama lain** (teks bebas). Nama kelas tiap cluster
disimpan di `session_state["class_maps"][run_id]` (`ui.runs.class_map_for`); tabel luas
(`area_table`), peta warna kelas, dan unduhan semuanya dibaca dari peta kelas itu.

Logika murni (tanpa Streamlit) ada di bagian atas modul dan diuji tanpa browser;
`render_labeling_panel` hanya menggambar.

Aturan yang tidak boleh diubah diam-diam:

  * **Nama lain mengalahkan Kelas.** Satu cluster hanya punya satu nama. Bila "Nama lain"
    terisi (setelah spasi tepi dibuang), itulah namanya; bila kosong, dipakai "Kelas"; bila
    keduanya kosong, `UNNAMED`. Aturan ini murni fungsi dari isi tabel yang tampak di layar,
    jadi hasilnya tidak bergantung pada urutan penyuntingan atau edit lama yang masih
    tersimpan di widget.
  * Isi editor dibangun ulang dari `class_map` pada tiap eksekusi: nama preset masuk ke
    Kelas; `UNNAMED` membuat keduanya kosong; nama lain masuk ke Nama lain. Dtype kedua
    kolom selalu `string` (bukan `object`/null), supaya skema tabel tidak berubah saat
    sel pertama diisi (skema menentukan identitas widget di Streamlit 1.67).
  * Kunci editor memuat `run_id`: dua run ber-K sama tidak berbagi state suntingan.
  * Nama yang ditolak (hanya `Total` yang bisa dari UI) mempertahankan nama sebelumnya
    dan ditampilkan sebagai galat di panel pada tiap eksekusi sampai diperbaiki.
  * Berkas unduhan dari hasil tiruan diberi awalan `TIRUAN_` pada NAMA BERKAS; isi CSV
    tidak ditambah kolom (CSV harus sama dengan tabel yang tampak).
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from landcover.contract import RunResult
from landcover.export import png_bytes, table_csv_bytes
from landcover.labeling import (
    CLASS_PRESETS, TOTAL_LABEL, UNNAMED, ClassMap, _normalized_class_map, area_table,
    cluster_table, normalize_class_name,
)
from landcover.render import class_palette, export_full_res_png, recolor
from ui.badges import MOCK_BADGE
from ui.compare import MOCK_NOTICE
from ui.runs import run_id_of

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
ROW_PX = 35                                # tinggi baris bawaan tabel Streamlit
TABLE_PAD_PX = 3


# --------------------------------------------------------------------------- #
# Logika murni
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Tampilan
# --------------------------------------------------------------------------- #

def _table_height(rows: int) -> int:
    return ROW_PX * (rows + 1) + TABLE_PAD_PX


def _editor_config() -> dict:
    cfg = st.column_config
    return {
        COL_COLOR: cfg.ImageColumn("Warna", help="Warna centroid: rata-rata RGB piksel cluster."),
        COL_CLUSTER: cfg.NumberColumn("Cluster", format="%d"),
        COL_PIXELS: cfg.NumberColumn("Piksel", format="%d"),
        COL_PERCENT: cfg.NumberColumn("Persen (%)", format="%.2f"),
        COL_CLASS: cfg.SelectboxColumn(
            "Kelas", options=list(CLASS_PRESETS), required=False,
            help="Cluster ber-Kelas sama digabung di tabel luas dan berwarna sama di peta.",
        ),
        COL_OTHER: cfg.TextColumn(
            "Nama lain", max_chars=40,
            help="Nama bebas, mis. Sawah. Bila diisi, mengalahkan kolom Kelas; kosongkan agar "
                 "pilihan Kelas dipakai. Huruf besar/kecil dibedakan. Nama 'Total' tidak boleh dipakai.",
        ),
    }


def _area_display_config() -> dict:
    cfg = st.column_config
    return {
        "kelas": cfg.TextColumn("Kelas"),
        "cluster": cfg.TextColumn("Cluster"),
        "piksel": cfg.NumberColumn("Piksel", format="%d"),
        "persen": cfg.NumberColumn("Persen (%)", format="%.2f"),
        "luas_ha": cfg.NumberColumn("Luas (ha)", format="%.2f"),
    }


def _render_downloads(run: RunResult, class_map: ClassMap) -> None:
    names = export_names(run_id_of(run), run.mock)
    left, middle, right = st.columns(3)
    try:
        map_png = class_map_png(run, class_map)
    except ValueError as exc:          # hasil janggal (centroid atau label di luar rentang)
        map_png = None
        left.caption(f"Peta kelas (PNG) tidak bisa dibuat: {exc}")
    if map_png is not None:
        left.download_button(
            "Peta kelas (PNG pratinjau)", data=map_png, file_name=names["map"],
            mime="image/png", key="dl_map", on_click="ignore", width="stretch",
        )
    middle.download_button(
        "Tabel luas per kelas (CSV)", data=area_csv(run, class_map), file_name=names["area"],
        mime="text/csv", key="dl_area_csv", on_click="ignore", width="stretch",
    )
    right.download_button(
        "Tabel per cluster (CSV)", data=cluster_csv(run), file_name=names["clusters"],
        mime="text/csv", key="dl_cluster_csv", on_click="ignore", width="stretch",
    )
    if run.mock:
        st.caption(f"Berkas unduhan hasil tiruan berawalan {MOCK_FILE_PREFIX}.")

    if labels_file_of(run) is None:
        st.caption(
            f"PNG resolusi penuh tidak tersedia: folder run ini tidak punya {LABELS_FILE} "
            "(dibuat oleh landcover --write-labels)."
        )
        return
    if st.button("Buat PNG resolusi penuh", key="btn_full_export",
                 help=f"Menulis PNG berpalet {run.width}×{run.height} ke folder run. Bisa beberapa detik."):
        try:
            with st.spinner("Membuat PNG resolusi penuh…"):
                target = export_full_map(run, class_map)
                data = target.read_bytes()
        except (OSError, ValueError) as exc:
            st.error(f"PNG resolusi penuh gagal dibuat: {exc}")
            return
        st.success(f"Disimpan di {target} ({len(data) / 1e6:.1f} MB).")
        st.download_button(
            "Unduh PNG resolusi penuh", data=data, file_name=names["full"], mime="image/png",
            key="dl_full", on_click="ignore",
        )


def render_labeling_panel(run: RunResult, class_map: ClassMap) -> bool:
    """Gambar editor, ringkasan, tabel luas, dan unduhan untuk `run`.

    Memperbarui `class_map` DI TEMPAT bila tabel editor berubah dan mengembalikan True bila
    berubah. Karena panel digambar setelah peta di skrip, pemanggil menggambar peta sesudahnya
    (lewat `st.container()` yang dibuat lebih awal) supaya peta memakai nama terbaru."""
    rid = run_id_of(run)
    size, size_note = pixel_size_of(run)

    st.subheader(f"Pelabelan cluster dan luas {MOCK_BADGE}" if run.mock else "Pelabelan cluster dan luas")
    st.caption(
        "Beri nama tiap cluster: pilih Kelas atau ketik Nama lain. Cluster bernama sama digabung, "
        "dan peta langsung memakai warna kelas."
    )
    if size_note:
        st.caption(size_note)

    try:
        frame = editor_frame(run.counts, run.centroids, class_map, size)
    except ValueError as exc:          # hasil janggal: counts/centroid tidak cocok, atau state basi
        st.error(f"Panel pelabelan tidak bisa dibuat untuk hasil ini: {exc}")
        return False
    k = len(frame)
    edited = st.data_editor(
        frame, key=editor_key(rid), hide_index=True, num_rows="fixed", width="stretch",
        height=_table_height(len(frame)), disabled=list(READ_ONLY_COLUMNS),
        column_order=list(EDITOR_COLUMNS), column_config=_editor_config(),
    )
    new_map, errors = class_map_from_frame(edited, class_map, k)
    changed = new_map != _normalized_class_map(class_map, k)
    if changed:
        class_map.clear()
        class_map.update(new_map)
    for message in errors:
        st.error(message)

    area = area_table(run.counts, class_map, size)
    summary = summary_of(area)
    st.markdown("**Luas per kelas**")
    # Dua baris dua kolom: nilai metrik tidak membungkus, jadi "1,205,604.00 ha" terpotong bila
    # empat metrik berdesakan dalam satu baris (terlihat di browser).
    top, bottom = st.columns(2), st.columns(2)
    top[0].metric("Luas total", format_ha(summary.total_ha))
    top[1].metric("Kelas bernama", str(summary.named_classes))
    bottom[0].metric(
        "Kelas terbesar",
        "Belum ada" if summary.largest is None else f"{summary.largest[0]} ({summary.largest[1]:.1f} %)",
    )
    bottom[1].metric("Belum diberi nama", format_percent(summary.unnamed_percent))
    st.dataframe(
        area, hide_index=True, width="stretch", height=_table_height(len(area)),
        column_config=_area_display_config(),
    )
    if run.mock:
        st.caption(MOCK_NOTICE)
    _render_downloads(run, class_map)
    return changed


def render_labeling_slot(run: RunResult | None = None, class_map: ClassMap | None = None) -> bool:
    """Kotak panel di halaman: panel bila ada run, selain itu kotak kosong dengan petunjuk.
    Mengembalikan True bila nama kelas berubah pada eksekusi ini."""
    with st.container(border=True):
        if run is None or class_map is None:
            st.caption("Pelabelan cluster dan tabel luas: belum ada hasil (jalankan dulu)")
            return False
        return render_labeling_panel(run, class_map)


__all__ = [
    "COL_CLASS", "COL_CLUSTER", "COL_COLOR", "COL_OTHER", "COL_PERCENT", "COL_PIXELS",
    "DEFAULT_PIXEL_SIZE_M", "EDITOR_COLUMNS", "KEY_EDITOR_PREFIX", "LABELS_FILE", "MOCK_FILE_PREFIX",
    "READ_ONLY_COLUMNS", "Summary", "area_csv", "class_map_from_frame", "class_map_png",
    "cluster_csv", "editor_frame", "editor_key", "export_full_map", "export_names", "format_ha",
    "format_percent", "labels_file_of", "pixel_size_of", "render_labeling_panel",
    "render_labeling_slot", "resolve_name", "split_name", "summary_of", "swatch_data_url",
]
