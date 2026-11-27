"""Logika murni panel pelabelan (plan Langkah 8), tanpa browser dan tanpa AppTest.

Alur layar (editor -> peta -> tabel) diuji di `test_app_flow.py`. Di sini: isi editor,
aturan "Nama lain mengalahkan Kelas", pemetaan tabel editor -> peta kelas, ringkasan,
nama berkas, CSV = tabel, PNG peta, dan ekspor resolusi penuh dari `labels.u8` buatan tangan.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from conftest import default_labels
from landcover.contract import read_run
from landcover.labeling import (
    CLASS_PRESETS, TOTAL_LABEL, UNNAMED, area_table, cluster_table, default_class_map,
)
from landcover.render import class_palette, recolor
from ui import labeling_panel as lp

CENTROIDS = np.array([[34, 61, 40], [201, 198, 190], [60, 120, 200], [180, 140, 60]], dtype=np.int64)
COUNTS = np.array([5, 0, 10, 1], dtype=np.int64)


def frame_for(class_map=None, counts=COUNTS, centroids=CENTROIDS, size=10.0):
    return lp.editor_frame(counts, centroids, default_class_map(len(counts)) if class_map is None else class_map, size)


def put(frame: pd.DataFrame, cluster: int, column: str, value) -> pd.DataFrame:
    out = frame.copy()
    row = out.index[out[lp.COL_CLUSTER] == cluster][0]
    if value is None or value is pd.NA:
        out[column] = out[column].astype("object")
        out.at[row, column] = None
        out[column] = out[column].astype("string")
    else:
        out.at[row, column] = value
    return out


# --- swatch ------------------------------------------------------------------ #

def decode_swatch(url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(url[len(prefix):])))


def colors_of(img: Image.Image) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in px) for px in np.array(img).reshape(-1, 3)}


def test_swatch_adalah_png_satu_warna_persis():
    img = decode_swatch(lp.swatch_data_url((12, 200, 34)))
    assert img.size == lp.SWATCH_SIZE and img.mode == "RGB"
    assert colors_of(img) == {(12, 200, 34)}


def test_swatch_warna_berbeda_url_berbeda_dan_stabil():
    assert lp.swatch_data_url((1, 2, 3)) == lp.swatch_data_url((1, 2, 3))
    assert lp.swatch_data_url((1, 2, 3)) != lp.swatch_data_url((1, 2, 4))


def test_swatch_memotong_nilai_di_luar_rentang_tanpa_galat():
    img = decode_swatch(lp.swatch_data_url((300, -5, 128)))
    assert colors_of(img) == {(255, 0, 128)}


# --- aturan nama --------------------------------------------------------------- #

@pytest.mark.parametrize("name", [UNNAMED, *CLASS_PRESETS, "Sawah", "Lahan basah", "air", "A"])
def test_split_lalu_resolve_mengembalikan_nama_yang_sama(name):
    assert lp.resolve_name(*lp.split_name(name)) == name


def test_split_preset_ke_kelas_dan_nama_bebas_ke_nama_lain():
    assert lp.split_name("Air") == ("Air", "")
    assert lp.split_name(UNNAMED) == (None, "")
    assert lp.split_name("Sawah") == (None, "Sawah")
    assert lp.split_name("  Sawah ") == (None, "Sawah")          # dirapikan dulu
    assert lp.split_name("air") == (None, "air")                  # huruf kecil bukan preset: nama bebas


def test_nama_lain_mengalahkan_kelas():
    assert lp.resolve_name("Air", "Sawah") == "Sawah"
    assert lp.resolve_name("Air", "  Sawah  ") == "Sawah"
    assert lp.resolve_name(None, "Sawah") == "Sawah"
    assert lp.resolve_name("Air", UNNAMED) == UNNAMED             # terisi (walau isinya teks bawaan) tetap menang


@pytest.mark.parametrize("blank", [None, pd.NA, float("nan"), "", "   ", "\t"])
def test_nama_lain_kosong_memakai_kelas(blank):
    assert lp.resolve_name("Air", blank) == "Air"
    assert lp.resolve_name(None, blank) == UNNAMED
    assert lp.resolve_name(pd.NA, blank) == UNNAMED
    assert lp.resolve_name(float("nan"), blank) == UNNAMED
    assert lp.resolve_name("", blank) == UNNAMED


@pytest.mark.parametrize("bad", ["Total", "total", " TOTAL "])
def test_nama_total_ditolak_di_kedua_kolom(bad):
    with pytest.raises(ValueError, match="Total"):
        lp.resolve_name(None, bad)
    with pytest.raises(ValueError, match="Total"):
        lp.resolve_name(bad, None)
    with pytest.raises(ValueError, match="Total"):
        lp.resolve_name("Air", bad)        # Nama lain menang, jadi galatnya tetap muncul


def test_nama_bukan_teks_ditolak():
    with pytest.raises(ValueError, match="teks"):
        lp.resolve_name(None, 5)
    with pytest.raises(ValueError, match="teks"):
        lp.resolve_name(["Air"], None)


# --- isi editor ----------------------------------------------------------------- #

def test_editor_frame_kolom_nilai_dan_warna():
    table = cluster_table(COUNTS, CENTROIDS, 10.0)
    frame = frame_for()
    assert list(frame.columns) == list(lp.EDITOR_COLUMNS)
    assert frame[lp.COL_CLUSTER].tolist() == [0, 1, 2, 3]
    assert frame[lp.COL_PIXELS].tolist() == [5, 0, 10, 1]
    assert frame[lp.COL_PERCENT].tolist() == table["persen"].tolist()     # sama persis dengan cluster_table
    for i, (r, g, b) in enumerate(CENTROIDS):
        assert colors_of(decode_swatch(frame[lp.COL_COLOR][i])) == {(int(r), int(g), int(b))}
    assert frame[lp.COL_CLASS].isna().all() and (frame[lp.COL_OTHER] == "").all()


def test_editor_frame_membagi_nama_ke_kelas_dan_nama_lain():
    frame = frame_for({0: "Air", 1: "Sawah", 2: UNNAMED, 3: "Lahan terbuka"})
    assert frame[lp.COL_CLASS].tolist()[0] == "Air" and pd.isna(frame[lp.COL_CLASS].tolist()[1])
    assert frame[lp.COL_OTHER].tolist() == ["", "Sawah", "", ""]
    assert pd.isna(frame[lp.COL_CLASS].tolist()[2]) and frame[lp.COL_CLASS].tolist()[3] == "Lahan terbuka"


def test_skema_editor_sama_walau_isinya_kosong_atau_terisi():
    """Skema (dtype/Arrow) menentukan identitas widget di Streamlit 1.67: tidak boleh berubah
    hanya karena sel pertama diisi."""
    empty = frame_for()
    filled = frame_for({0: "Air", 1: "Sawah", 2: "Vegetasi", 3: "Lahan terbangun"})
    assert list(empty.dtypes) == list(filled.dtypes)
    assert str(empty[lp.COL_CLASS].dtype) == "string" and str(empty[lp.COL_OTHER].dtype) == "string"
    assert pa.Schema.from_pandas(empty) == pa.Schema.from_pandas(filled)
    assert not pa.types.is_null(pa.Schema.from_pandas(empty).field(lp.COL_CLASS).type)


def test_editor_frame_menolak_state_basi_dan_masukan_salah():
    with pytest.raises(ValueError, match="di luar rentang"):
        frame_for({**default_class_map(4), 9: "Air"})                  # id cluster dari run lain
    with pytest.raises(ValueError):
        frame_for(counts=np.array([1.0, 2.0]), centroids=CENTROIDS[:2])
    with pytest.raises(ValueError, match="jumlah centroid"):
        frame_for(centroids=CENTROIDS[:3])


def test_editor_frame_cluster_berjumlah_nol_tampil_nol_persen():
    frame = frame_for()
    assert frame.loc[frame[lp.COL_CLUSTER] == 1, lp.COL_PIXELS].item() == 0
    assert frame.loc[frame[lp.COL_CLUSTER] == 1, lp.COL_PERCENT].item() == 0.0


# --- tabel editor -> peta kelas ---------------------------------------------------- #

@pytest.mark.parametrize("class_map", [
    default_class_map(4),
    {0: "Air", 1: "Air", 2: "Sawah", 3: UNNAMED},
    {0: "Vegetasi", 1: "Awan/bayangan", 2: "air", 3: "Lahan terbuka"},
    {0: " Air ", 2: "Sawah"},                       # kunci hilang = UNNAMED; spasi dirapikan
])
def test_tabel_tanpa_suntingan_tidak_mengubah_peta(class_map):
    new, errors = lp.class_map_from_frame(frame_for(class_map), class_map, 4)
    assert errors == []
    assert new == {i: lp.normalize_class_name(class_map.get(i)) for i in range(4)}


def test_gabungkan_dua_cluster_ke_air_menjadi_satu_baris_air():
    frame = put(put(frame_for(), 0, lp.COL_CLASS, "Air"), 2, lp.COL_CLASS, "Air")
    new, errors = lp.class_map_from_frame(frame, default_class_map(4), 4)
    assert errors == [] and new == {0: "Air", 1: UNNAMED, 2: "Air", 3: UNNAMED}
    area = area_table(COUNTS, new, 10.0)
    air = area[area["kelas"] == "Air"]
    assert len(air) == 1
    assert air["piksel"].item() == 15 and air["cluster"].item() == "0, 2"
    assert area["kelas"].tolist() == ["Air", UNNAMED, TOTAL_LABEL]
    assert area["piksel"].iloc[-1] == COUNTS.sum()


def test_nama_lain_dan_kelas_pada_baris_yang_sama():
    frame = put(put(frame_for(), 0, lp.COL_CLASS, "Air"), 0, lp.COL_OTHER, "Sawah")
    new, _ = lp.class_map_from_frame(frame, default_class_map(4), 4)
    assert new[0] == "Sawah"
    frame = put(frame, 0, lp.COL_OTHER, "")                          # Nama lain dikosongkan -> Kelas dipakai
    new, _ = lp.class_map_from_frame(frame, default_class_map(4), 4)
    assert new[0] == "Air"
    frame = put(frame, 0, lp.COL_CLASS, None)                        # Kelas dikosongkan juga -> tanpa nama
    new, _ = lp.class_map_from_frame(frame, default_class_map(4), 4)
    assert new[0] == UNNAMED


def test_cluster_dipetakan_lewat_kolom_cluster_bukan_posisi_baris():
    frame = put(frame_for(), 3, lp.COL_CLASS, "Air")
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    new, _ = lp.class_map_from_frame(shuffled, default_class_map(4), 4)
    assert new == {0: UNNAMED, 1: UNNAMED, 2: UNNAMED, 3: "Air"}


def test_nama_total_mempertahankan_nama_sebelumnya_dan_sisanya_tetap_berlaku():
    previous = {0: "Vegetasi", 1: UNNAMED, 2: UNNAMED, 3: UNNAMED}
    frame = put(put(frame_for(previous), 0, lp.COL_OTHER, "Total"), 2, lp.COL_CLASS, "Air")
    new, errors = lp.class_map_from_frame(frame, previous, 4)
    assert new[0] == "Vegetasi" and new[2] == "Air"                  # cluster 0 tidak berubah, cluster 2 berlaku
    assert len(errors) == 1 and errors[0].startswith("Cluster 0:") and "Total" in errors[0]
    assert "Vegetasi" in errors[0]                                    # pesan menyebut nama yang dipertahankan


def test_peta_sebelumnya_tidak_diubah_oleh_fungsi():
    previous = {0: "Air", 1: UNNAMED, 2: UNNAMED, 3: UNNAMED}
    snapshot = dict(previous)
    class_map_new, _ = lp.class_map_from_frame(put(frame_for(previous), 1, lp.COL_CLASS, "Air"), previous, 4)
    assert previous == snapshot and class_map_new != previous


def test_kolom_hilang_dan_cluster_di_luar_rentang_adalah_valueerror():
    with pytest.raises(ValueError, match="kolom editor hilang"):
        lp.class_map_from_frame(frame_for().drop(columns=[lp.COL_OTHER]), default_class_map(4), 4)
    frame = frame_for()
    frame.loc[0, lp.COL_CLUSTER] = 7
    with pytest.raises(ValueError, match="di luar rentang"):
        lp.class_map_from_frame(frame, default_class_map(4), 4)


def test_sel_kosong_dari_editor_jadi_tanpa_nama():
    frame = frame_for({0: "Air", 1: "Air", 2: "Air", 3: "Air"})
    frame = put(frame, 1, lp.COL_CLASS, None)                        # SelectboxColumn dikosongkan (NA)
    frame[lp.COL_OTHER] = frame[lp.COL_OTHER].astype("object")
    frame.at[2, lp.COL_CLASS] = None
    frame.at[2, lp.COL_OTHER] = float("nan")
    new, errors = lp.class_map_from_frame(frame, default_class_map(4), 4)
    assert errors == [] and new == {0: "Air", 1: UNNAMED, 2: UNNAMED, 3: "Air"}


# --- ringkasan ---------------------------------------------------------------------- #

def test_ringkasan_semua_belum_bernama():
    s = lp.summary_of(area_table(COUNTS, default_class_map(4), 10.0))
    assert s.named_classes == 0 and s.largest is None
    assert s.unnamed_percent == 100.0
    assert s.total_ha == pytest.approx(16 * 100 / 10_000)


def test_ringkasan_campuran_dan_kelas_terbesar():
    area = area_table(COUNTS, {0: "Air", 1: "Air", 2: "Sawah", 3: UNNAMED}, 10.0)
    s = lp.summary_of(area)
    assert s.named_classes == 2
    assert s.largest == ("Sawah", pytest.approx(10 * 100 / 16))
    assert s.unnamed_percent == pytest.approx(1 * 100 / 16)


def test_ringkasan_gabungan_bisa_mengalahkan_cluster_tunggal_terbesar():
    area = area_table(COUNTS, {0: "Air", 1: UNNAMED, 2: "Sawah", 3: "Air"}, 10.0)     # Air = 5 + 1 = 6 < Sawah 10
    assert lp.summary_of(area).largest[0] == "Sawah"
    area = area_table(np.array([6, 5, 10, 0]), {0: "Air", 1: "Air", 2: "Sawah", 3: UNNAMED}, 10.0)
    assert lp.summary_of(area).largest == ("Air", pytest.approx(11 * 100 / 21))       # 11 > 10: gabungan menang


def test_ringkasan_tile_penuh_tepat():
    n = 10980 * 10980
    area = area_table(np.array([n - 1000, 1000]), {0: "Vegetasi", 1: "Air"}, 10.0)
    s = lp.summary_of(area)
    assert s.total_ha == 1_205_604.0 and s.named_classes == 2 and s.unnamed_percent == 0.0


def test_format_angka_tampilan():
    assert lp.format_ha(1_205_604.0) == "1,205,604.00 ha"
    assert lp.format_ha(12.0) == "12.00 ha"
    assert lp.format_percent(100.0) == "100.00 %"


# --- ukuran piksel -------------------------------------------------------------------- #

def test_pixel_size_dari_meta_dan_cadangannya(make_run_dir):
    run = read_run(make_run_dir(meta={"pixel_size_m": 20}))
    assert lp.pixel_size_of(run) == (20.0, None)
    run = read_run(make_run_dir(meta={"pixel_size_m": None}))                # kunci dihapus
    size, note = lp.pixel_size_of(run)
    assert size == 10.0 and "tidak mencatat" in note
    for bad in (0, -3, "10", True, float("nan"), [10]):
        run = read_run(make_run_dir(meta={"pixel_size_m": bad}))
        size, note = lp.pixel_size_of(run)
        assert size == 10.0 and "tidak valid" in note, bad


# --- unduhan ----------------------------------------------------------------------------- #

def test_nama_berkas_unduhan_tiruan_berawalan_dan_asli_tidak():
    mock = lp.export_names("kecil__cpu__k8_it20_s42_t1", True)
    real = lp.export_names("kecil__cpu__k8_it20_s42_t1", False)
    assert set(mock) == set(real) == {"map", "area", "clusters", "full"}
    assert all(name.startswith("TIRUAN_") for name in mock.values())
    assert not any("TIRUAN" in name for name in real.values())
    assert all("kecil__cpu__k8_it20_s42_t1" in name for name in (*mock.values(), *real.values()))
    assert {n.rsplit(".", 1)[1] for n in real.values()} == {"png", "csv"}
    assert len(set(real.values())) == 4                                       # keempat nama berbeda


def test_csv_luas_sama_dengan_tabel_yang_ditampilkan(make_run_dir):
    run = read_run(make_run_dir(k=4, counts=[5, 0, 10, 1]))
    cmap = {0: "Air", 1: "Air", 2: "Sawah", 3: UNNAMED}
    table = area_table(run.counts, cmap, 10.0)                               # persis yang digambar panel
    csv = pd.read_csv(io.BytesIO(lp.area_csv(run, cmap)), dtype=str, keep_default_na=False)
    assert list(csv.columns) == ["kelas", "cluster", "piksel", "persen", "luas_ha"]
    assert csv["kelas"].tolist() == table["kelas"].tolist() == ["Air", "Sawah", UNNAMED, TOTAL_LABEL]
    assert csv["cluster"].tolist() == table["cluster"].tolist() == ["0, 1", "2", "3", ""]
    assert csv["piksel"].tolist() == [str(v) for v in table["piksel"]]
    assert csv["persen"].tolist() == [f"{v:.2f}" for v in table["persen"]]
    assert csv["luas_ha"].tolist() == [f"{v:.2f}" for v in table["luas_ha"]]
    assert csv["piksel"].iloc[0] == "5"                                      # 5 + 0 piksel
    assert csv["persen"].iloc[-1] == "100.00"


def test_csv_luas_memakai_ukuran_piksel_meta(make_run_dir):
    run = read_run(make_run_dir(k=2, counts=[100, 100], meta={"pixel_size_m": 20}))
    csv = pd.read_csv(io.BytesIO(lp.area_csv(run, {0: "Air", 1: "Air"})), dtype=str, keep_default_na=False)
    assert csv["kelas"].tolist() == ["Air", TOTAL_LABEL]
    assert csv["luas_ha"].tolist() == ["8.00", "8.00"]                       # 200 piksel * 20^2 m^2 / 10 000 = 8 ha


def test_csv_per_cluster_header_dan_isi(make_run_dir):
    run = read_run(make_run_dir(k=4, counts=[5, 0, 10, 1]))
    csv = pd.read_csv(io.BytesIO(lp.cluster_csv(run)), dtype=str, keep_default_na=False)
    assert list(csv.columns) == ["cluster", "r", "g", "b", "piksel", "persen", "luas_ha"]
    assert csv["piksel"].tolist() == ["5", "0", "10", "1"]
    assert csv["r"].tolist() == [str(v) for v in run.centroids[:, 0]]


def test_png_peta_kelas_ukuran_pratinjau_dan_warna_gabungan_sama(make_run_dir):
    run = read_run(make_run_dir(width=8, height=5, k=4))
    cmap = {0: "Air", 1: "Air", 2: UNNAMED, 3: "Sawah"}
    img = np.array(Image.open(io.BytesIO(lp.class_map_png(run, cmap))))
    assert img.shape == (5, 8, 3)
    expected = recolor(run.labels_preview, class_palette(cmap, run.centroids))
    assert np.array_equal(img, expected)
    air = tuple(class_palette(cmap, run.centroids)[0])
    assert tuple(class_palette(cmap, run.centroids)[1]) == air               # cluster gabungan: satu warna
    unnamed = tuple(class_palette(cmap, run.centroids)[2])
    assert unnamed == tuple(int(c) for c in run.centroids[2])                # tanpa nama: warna centroid


def test_png_peta_kelas_menolak_label_di_luar_palet(make_run_dir):
    run = read_run(make_run_dir(width=4, height=2, k=4))
    run.labels_preview[0, 0] = 9
    with pytest.raises(ValueError, match="di luar rentang"):
        lp.class_map_png(run, default_class_map(4))


# --- ekspor resolusi penuh ------------------------------------------------------------------- #

def make_full_run(make_run_dir, tmp_path, *, w=6, h=4, k=3, mock=True, labels_bytes=None):
    """Run dengan pratinjau 3x2 tetapi ukuran penuh w x h (meta), plus `labels.u8` buatan tangan."""
    run_dir = make_run_dir(width=3, height=2, k=k, labels=default_labels(3, 2, k), meta={"width": w, "height": h, "mock": mock})
    full = (np.arange(w * h) % k).astype(np.uint8)
    (run_dir / lp.LABELS_FILE).write_bytes(full.tobytes() if labels_bytes is None else labels_bytes)
    return read_run(run_dir), full.reshape(h, w)


def test_ekspor_penuh_memakai_ukuran_meta_bukan_pratinjau(make_run_dir, tmp_path):
    run, full = make_full_run(make_run_dir, tmp_path)
    assert (run.width, run.height) == (6, 4) and run.labels_preview.shape == (2, 3)
    cmap = {0: "Air", 1: "Air", 2: "Sawah"}
    out = lp.export_full_map(run, cmap)
    assert out == run.path / lp.export_names(run.path.name, True)["full"] and out.name.startswith("TIRUAN_")
    with Image.open(out) as png:
        assert png.mode == "P" and png.size == (6, 4)                      # lebar 6, tinggi 4
        assert np.array_equal(np.array(png), full)                         # indeks = id cluster, bukan transpos
        palette = np.array(png.getpalette()[:9], dtype=np.uint8).reshape(3, 3)
    assert np.array_equal(palette, class_palette(cmap, run.centroids))
    assert not list(run.path.glob("*.tmp"))


def test_ekspor_penuh_run_asli_tanpa_awalan_dan_ke_jalur_pilihan(make_run_dir, tmp_path):
    run, full = make_full_run(make_run_dir, tmp_path, mock=False)
    out = lp.export_full_map(run, default_class_map(3))
    assert not out.name.startswith("TIRUAN_")
    target = tmp_path / "keluar" / "peta.png"
    assert lp.export_full_map(run, default_class_map(3), target) == target and target.is_file()


def test_ekspor_penuh_tanpa_labels_u8_filenotfound(make_run_dir):
    run = read_run(make_run_dir())
    assert lp.labels_file_of(run) is None
    with pytest.raises(FileNotFoundError, match="labels.u8"):
        lp.export_full_map(run, default_class_map(4))


def test_ekspor_penuh_ukuran_salah_atau_label_di_luar_palet_tidak_menulis_apa_apa(make_run_dir, tmp_path):
    run, _ = make_full_run(make_run_dir, tmp_path, labels_bytes=bytes(23))      # seharusnya 24 byte
    with pytest.raises(ValueError, match="24"):
        lp.export_full_map(run, default_class_map(3))
    assert not list(run.path.glob("*.png.tmp")) and not (run.path / lp.export_names(run.path.name, True)["full"]).exists()

    run, _ = make_full_run(make_run_dir, tmp_path, labels_bytes=bytes([0, 1, 2, 3] * 6))   # label 3 >= K=3
    with pytest.raises(ValueError, match="di luar palet"):
        lp.export_full_map(run, default_class_map(3))
    assert not (run.path / lp.export_names(run.path.name, True)["full"]).exists()
