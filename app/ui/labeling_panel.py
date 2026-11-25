"""Panel pelabelan cluster dan tabel luas (plan Langkah 8).

Satu baris editor per cluster: contoh warna centroid, id, piksel, persen, kolom **Kelas**
(pilihan `CLASS_PRESETS`) dan kolom **Nama lain** (teks bebas). Nama kelas tiap cluster
disimpan di `session_state["class_maps"][run_id]` (`ui.runs.class_map_for`); tabel luas
(`area_table`), peta warna kelas, dan unduhan semuanya dibaca dari peta kelas itu.

Logika murni (nama kelas, ringkasan, CSV, PNG, nama berkas) ada di `services/labeling_panel.py`
(dipakai juga oleh API FastAPI) dan diuji tanpa browser; modul ini mengimpornya, jadi nama-nama
lama tetap berlaku. `render_labeling_panel` hanya menggambar.

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

import streamlit as st

from landcover.contract import RunResult
from landcover.labeling import (
    CLASS_PRESETS, ClassMap, _normalized_class_map, area_table, normalize_class_name,
)
from services.compare import MOCK_NOTICE
from services.labeling_panel import (
    COL_CLASS,
    COL_CLUSTER,
    COL_COLOR,
    COL_OTHER,
    COL_PERCENT,
    COL_PIXELS,
    DEFAULT_PIXEL_SIZE_M,
    EDITOR_COLUMNS,
    EXPORT_LABELS,
    HELP_CLASS,
    HELP_OTHER,
    HELP_PANEL,
    KEY_EDITOR_PREFIX,
    LABELS_FILE,
    MOCK_FILE_PREFIX,
    READ_ONLY_COLUMNS,
    SWATCH_SIZE,
    Summary,
    area_csv,
    class_map_from_frame,
    class_map_png,
    cluster_csv,
    editor_frame,
    editor_key,
    export_full_map,
    export_names,
    format_ha,
    format_percent,
    labels_file_of,
    nodata_note,
    pixel_size_of,
    resolve_name,
    split_name,
    summary_of,
    swatch_data_url,
)
from services.runs import run_id_of
from ui.badges import MOCK_BADGE

ROW_PX = 35                                # tinggi baris bawaan tabel Streamlit
TABLE_PAD_PX = 3


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
            "Kelas", options=list(CLASS_PRESETS), required=False, help=HELP_CLASS,
        ),
        COL_OTHER: cfg.TextColumn("Nama lain", max_chars=40, help=HELP_OTHER),
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
            EXPORT_LABELS["map"], data=map_png, file_name=names["map"],
            mime="image/png", key="dl_map", on_click="ignore", width="stretch",
        )
    middle.download_button(
        EXPORT_LABELS["area"], data=area_csv(run, class_map), file_name=names["area"],
        mime="text/csv", key="dl_area_csv", on_click="ignore", width="stretch",
    )
    right.download_button(
        EXPORT_LABELS["clusters"], data=cluster_csv(run), file_name=names["clusters"],
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
    st.caption(HELP_PANEL)
    if size_note:
        st.caption(size_note)

    try:
        frame = editor_frame(
            run.counts, run.centroids, class_map, size,
            nodata_pixels=run.nodata_pixels, nodata_cluster=run.nodata_cluster,
        )
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

    area = area_table(
        run.counts, class_map, size,
        nodata_pixels=run.nodata_pixels, nodata_cluster=run.nodata_cluster,
    )
    summary = summary_of(area)
    nd_msg = nodata_note(run.nodata_pixels, int(run.counts.sum()))
    if nd_msg:
        st.info(nd_msg)
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
