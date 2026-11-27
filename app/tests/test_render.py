"""Tes `landcover.render` dan `landcover.export` (plan Langkah 5).

Satu file untuk kedua modul (plan: "Cek: pytest app/tests/test_render.py"): palet kelas,
pewarnaan ulang, PNG bytes, CSV bytes, dan ekspor PNG resolusi penuh.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from landcover.contract import read_run, validate
from landcover.export import CSV_DECIMALS, png_bytes, table_csv_bytes
from landcover.labeling import (
    CLASS_PRESETS,
    UNNAMED,
    area_table,
    cluster_table,
    default_class_map,
)
from landcover.render import (
    CLASS_COLORS,
    class_palette,
    export_full_res_png,
    fallback_color,
    recolor,
)
from launchers.base import RunParams, out_dir_for
from launchers.mock import MockLauncher

APP_DIR = Path(__file__).resolve().parents[1]

RGB4 = np.array([[34, 61, 40], [201, 198, 190], [60, 120, 200], [180, 140, 60]], dtype=np.int64)
RGB6 = np.array(
    [[34, 61, 40], [201, 198, 190], [60, 120, 200], [180, 140, 60], [90, 160, 70], [120, 120, 120]],
    dtype=np.int64,
)


def colors_of(palette: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(c) for c in row) for row in palette]


# --------------------------------------------------------------------------- #
# CLASS_COLORS dan fallback_color
# --------------------------------------------------------------------------- #

def test_class_colors_persis_sesuai_plan():
    assert CLASS_COLORS == {
        "Vegetasi": (34, 139, 34), "Air": (30, 100, 200), "Lahan terbangun": (200, 60, 60),
        "Lahan terbuka": (210, 180, 120), "Awan/bayangan": (170, 170, 170), "Lainnya": (90, 90, 90),
    }


def test_class_colors_kunci_sama_dengan_presets_dan_warna_berbeda():
    assert list(CLASS_COLORS) == CLASS_PRESETS                  # kunci dan urutan
    assert UNNAMED not in CLASS_COLORS
    assert len(set(CLASS_COLORS.values())) == len(CLASS_COLORS)  # tidak ada dua kelas satu warna
    for color in CLASS_COLORS.values():
        assert len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color)


SAMPLE_NAMES = ["Sawah", "Kebun", "Tambak", "Hutan", "sawah", "Sawah ", "Mangrove", "Pasir"]


def test_fallback_color_deterministik_dan_valid():
    for name in SAMPLE_NAMES:
        color = fallback_color(name)
        assert color == fallback_color(name)
        assert len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        assert max(color) - min(color) >= 60                    # bukan abu-abu
    # nama persis sama => warna sama; huruf besar/kecil dibedakan (seperti penggabungan di labeling)
    assert fallback_color("Sawah") != fallback_color("sawah")


def test_fallback_color_nama_contoh_berbeda_dan_tidak_menabrak_preset():
    found = [fallback_color(n) for n in ["Sawah", "Kebun", "Tambak", "Hutan", "Mangrove", "Pasir"]]
    assert len(set(found)) == len(found)
    assert not set(found) & set(CLASS_COLORS.values())


def test_fallback_color_nilai_acuan():
    # Dihitung terpisah dari kode: sha256("Sawah")[:4] = 0d3598bf = 221616319, mod 360 = hue 319;
    # HSV (S = 0,6, V = 0,85): C = 0,51, X = 0,2023, m = 0,34 -> (R, G, B) = (C+m, m, X+m) = (217, 87, 176).
    # Bila ini berubah, warna kelas isian bebas di laporan dan video ikut berubah.
    assert fallback_color("Sawah") == (217, 87, 176)


def test_fallback_color_tidak_bergantung_pada_hash_acak_python():
    # Mutan `hash(name)` memberi warna berbeda untuk PYTHONHASHSEED berbeda. Proses anak memakai `-P`
    # (cwd tidak masuk sys.path) dan BUKAN `-I`: `-I` mengabaikan PYTHONHASHSEED sehingga seed di bawah sia-sia.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from landcover.render import fallback_color; print(fallback_color('Sawah'), fallback_color('Kebun'))"
    )
    outputs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-P", "-c", code, str(APP_DIR)],
            capture_output=True, text=True, env=env, check=True,
        )
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1
    assert outputs == {f"{fallback_color('Sawah')} {fallback_color('Kebun')}"}


# --------------------------------------------------------------------------- #
# class_palette
# --------------------------------------------------------------------------- #

def test_palette_bentuk_dan_tipe():
    pal = class_palette(default_class_map(4), RGB4)
    assert pal.shape == (4, 3) and pal.dtype == np.uint8


def test_palette_semua_belum_bernama_memakai_warna_centroid():
    pal = class_palette(default_class_map(4), RGB4)
    assert np.array_equal(pal, RGB4.astype(np.uint8))
    assert np.array_equal(class_palette({}, RGB4), RGB4.astype(np.uint8))   # kosong = semua UNNAMED


def test_palette_kelas_preset_memakai_class_colors_dan_unnamed_memakai_centroid():
    pal = class_palette({0: "Vegetasi", 1: "Air", 3: "Lahan terbuka"}, RGB4)
    assert colors_of(pal) == [
        CLASS_COLORS["Vegetasi"], CLASS_COLORS["Air"], tuple(RGB4[2]), CLASS_COLORS["Lahan terbuka"],
    ]


def test_palette_cluster_digabung_berwarna_sama():
    # Preset yang diulang: centroid kedua cluster berbeda, warna kelas harus sama.
    pal = class_palette({0: "Air", 2: "Air", 1: "Vegetasi"}, RGB4)
    assert tuple(pal[0]) == tuple(pal[2]) == CLASS_COLORS["Air"]
    assert tuple(pal[0]) != tuple(RGB4[0]) and tuple(pal[2]) != tuple(RGB4[2])


def test_palette_isian_bebas_memakai_fallback_dan_bukan_warna_centroid():
    pal = class_palette({0: "Sawah", 3: "Sawah", 1: "Kebun"}, RGB4)
    assert tuple(pal[0]) == tuple(pal[3]) == fallback_color("Sawah")   # digabung: warna sama
    assert tuple(pal[1]) == fallback_color("Kebun")
    assert tuple(pal[2]) == tuple(RGB4[2])                              # UNNAMED tetap centroid
    assert tuple(pal[0]) != tuple(RGB4[0]) and tuple(pal[3]) != tuple(RGB4[3])
    assert tuple(pal[0]) != tuple(pal[1])


def test_palette_konsisten_dengan_area_table_untuk_peta_campuran():
    # Setiap baris kelas di tabel luas = satu warna di peta (kecuali UNNAMED: warna centroid).
    k = 6
    class_map = {0: "Air", 1: "Sawah", 2: "Air", 3: "Sawah", 4: "Lahan terbangun", 5: UNNAMED}
    counts = np.array([10, 20, 30, 40, 50, 60])
    pal = class_palette(class_map, RGB6)
    table = area_table(counts, class_map)
    seen = {}
    for _, row in table.iloc[:-1].iterrows():
        ids = [int(i) for i in row["cluster"].split(", ")]
        if row["kelas"] == UNNAMED:
            for i in ids:
                assert tuple(pal[i]) == tuple(RGB6[i])
        else:
            assert len({tuple(pal[i]) for i in ids}) == 1, row["kelas"]
            seen[row["kelas"]] = tuple(pal[ids[0]])
    assert set(seen) == {"Air", "Sawah", "Lahan terbangun"}
    assert len(set(seen.values())) == 3                                 # kelas berbeda, warna berbeda
    assert len(pal) == k


def test_palette_nama_dirapikan_seperti_labeling():
    pal = class_palette({0: "  Air  ", 1: "Air", 2: "", 3: None}, RGB4)
    assert tuple(pal[0]) == tuple(pal[1]) == CLASS_COLORS["Air"]
    assert tuple(pal[2]) == tuple(RGB4[2]) and tuple(pal[3]) == tuple(RGB4[3])   # kosong/None = UNNAMED
    nan_map = {0: float("nan"), 1: pd.NA}
    pal2 = class_palette(nan_map, RGB4)
    assert tuple(pal2[0]) == tuple(RGB4[0]) and tuple(pal2[1]) == tuple(RGB4[1])


def test_palette_kunci_numpy_diterima_dan_kunci_salah_ditolak():
    pal = class_palette({np.int64(1): "Air"}, RGB4)
    assert tuple(pal[1]) == CLASS_COLORS["Air"]
    with pytest.raises(ValueError, match="di luar rentang"):
        class_palette({4: "Air"}, RGB4)                                 # kunci basi dari run lain
    with pytest.raises(ValueError, match="di luar rentang"):
        class_palette({-1: "Air"}, RGB4)
    with pytest.raises(ValueError, match="harus integer"):
        class_palette({"0": "Air"}, RGB4)


def test_palette_nama_total_dan_non_teks_ditolak():
    with pytest.raises(ValueError, match="Total"):
        class_palette({0: "Total"}, RGB4)
    with pytest.raises(ValueError, match="TOTAL|Total"):
        class_palette({0: " total "}, RGB4)
    with pytest.raises(ValueError, match="teks"):
        class_palette({0: 3}, RGB4)


@pytest.mark.parametrize("bad", [
    [[34, 61, 256]], [[-1, 0, 0]], [[0, 0, 0], [300, 0, 0]],
])
def test_palette_centroid_di_luar_0_255_ditolak_juga_bila_semua_bernama(bad):
    k = len(bad)
    with pytest.raises(ValueError, match="0..255"):
        class_palette({i: "Air" for i in range(k)}, np.array(bad, dtype=np.int64))


def test_palette_centroid_bentuk_atau_tipe_salah_ditolak():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        class_palette({}, np.zeros((4, 2), dtype=np.int64))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        class_palette({}, np.zeros((0, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="integer"):
        class_palette({}, RGB4.astype(np.float64))
    with pytest.raises(ValueError, match="integer"):
        class_palette({}, np.ones((2, 3), dtype=bool))


def test_palette_warna_kustom_menimpa_dan_divalidasi():
    custom = {"Air": (1, 2, 3), "Sawah": [4, 5, 6]}
    pal = class_palette({0: "Air", 1: "Sawah", 2: "Vegetasi"}, RGB4, colors=custom)
    assert tuple(pal[0]) == (1, 2, 3) and tuple(pal[1]) == (4, 5, 6)
    assert tuple(pal[2]) == fallback_color("Vegetasi")                  # tidak ada di colors => fallback
    for bad in [(256, 0, 0), (0, 0), (0, 0, 0, 0), (-1, 0, 0), (1.5, 2, 3), (True, 0, 0), "abc", None]:
        with pytest.raises(ValueError, match="warna"):
            class_palette({0: "Air"}, RGB4, colors={"Air": bad})


def test_palette_warna_tidak_terpakai_tidak_divalidasi_dan_unnamed_di_colors_diabaikan():
    colors = {**CLASS_COLORS, "Tidak dipakai": (999, 0, 0), UNNAMED: (1, 2, 3)}
    pal = class_palette({0: "Air"}, RGB4, colors=colors)
    assert tuple(pal[0]) == CLASS_COLORS["Air"]
    assert tuple(pal[1]) == tuple(RGB4[1])                              # UNNAMED di colors tidak berlaku


def test_palette_tidak_mengubah_masukan():
    class_map = {0: " Air ", 1: "Sawah"}
    centroids = RGB4.copy()
    class_palette(class_map, centroids)
    assert class_map == {0: " Air ", 1: "Sawah"} and np.array_equal(centroids, RGB4)


def test_palette_k_16_dan_k_1():
    cen16 = np.arange(48, dtype=np.int64).reshape(16, 3)
    pal16 = class_palette({i: "Air" for i in range(0, 16, 2)}, cen16)
    assert pal16.shape == (16, 3)
    assert {tuple(pal16[i]) for i in range(0, 16, 2)} == {CLASS_COLORS["Air"]}
    assert np.array_equal(pal16[1::2], cen16[1::2].astype(np.uint8))
    assert class_palette({0: "Lainnya"}, RGB4[:1]).tolist() == [list(CLASS_COLORS["Lainnya"])]


# --------------------------------------------------------------------------- #
# recolor
# --------------------------------------------------------------------------- #

def test_recolor_persis_palette_indeks_label():
    pal = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    labels = np.array([[0, 1, 2, 2], [2, 1, 0, 0]], dtype=np.uint8)     # (2, 4), tidak persegi
    out = recolor(labels, pal)
    assert out.shape == (2, 4, 3) and out.dtype == np.uint8
    assert out[0, 0].tolist() == [10, 20, 30] and out[0, 3].tolist() == [70, 80, 90]
    assert out[1, 1].tolist() == [40, 50, 60] and out[1, 2].tolist() == [10, 20, 30]
    assert np.array_equal(out, pal[labels])


def test_recolor_tipe_label_integer_apa_pun_dan_hasil_salinan():
    pal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    base = np.array([[0, 1], [1, 0]])
    for dtype in (np.uint8, np.int32, np.int64, np.uint16):
        assert np.array_equal(recolor(base.astype(dtype), pal), pal[base])
    pal_before, labels = pal.copy(), base.astype(np.uint8)
    out = recolor(labels, pal)
    out[:] = 99
    assert np.array_equal(pal, pal_before) and np.array_equal(labels, base)   # masukan utuh


def test_recolor_palet_int64_diterima_selama_0_255():
    out = recolor(np.zeros((2, 2), dtype=np.uint8), np.array([[255, 0, 7]], dtype=np.int64))
    assert out.dtype == np.uint8 and out[1, 1].tolist() == [255, 0, 7]


def test_recolor_label_di_luar_palet_ditolak_dengan_pesan_jelas():
    pal = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"0\.\.2.*max 3"):
        recolor(np.array([[0, 3]], dtype=np.uint8), pal)
    with pytest.raises(ValueError, match="min -1"):
        recolor(np.array([[-1, 0]], dtype=np.int64), pal)
    with pytest.raises(ValueError, match="max 200"):
        recolor(np.array([[200]], dtype=np.uint8), pal)


@pytest.mark.parametrize("bad", [
    np.array([[0.0, 1.0]]), np.array([[True, False]]), np.zeros(3, dtype=np.uint8),
    np.zeros((2, 2, 3), dtype=np.uint8),
])
def test_recolor_label_bentuk_atau_tipe_salah_ditolak(bad):
    with pytest.raises(ValueError, match="labels"):
        recolor(bad, np.zeros((2, 3), dtype=np.uint8))


@pytest.mark.parametrize("bad", [
    np.zeros((2, 2), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8),
    np.full((2, 3), 256, dtype=np.int64), np.zeros((2, 3)),
])
def test_recolor_palet_salah_ditolak(bad):
    with pytest.raises(ValueError, match="palette"):
        recolor(np.zeros((2, 2), dtype=np.uint8), bad)


def test_recolor_gambar_kosong_tidak_error():
    assert recolor(np.zeros((0, 5), dtype=np.uint8), np.zeros((1, 3), dtype=np.uint8)).shape == (0, 5, 3)


def test_recolor_run_nyata_tidak_persegi_ukuran_sama_dengan_pratinjau(make_run_dir):
    run = read_run(make_run_dir(width=37, height=19, k=5))
    assert run.labels_preview.shape == (19, 37)
    class_map = {0: "Air", 1: "Air", 2: "Sawah"}
    out = recolor(run.labels_preview, class_palette(class_map, run.centroids))
    assert out.shape == run.labels_preview.shape + (3,) == run.map_preview.shape == (19, 37, 3)
    # Gabungan cluster 0 dan 1 satu warna: tepat itu yang ada di peta di piksel berlabel 0 atau 1.
    merged = np.isin(run.labels_preview, [0, 1])
    assert merged.any() and np.all(out[merged] == np.array(CLASS_COLORS["Air"], dtype=np.uint8))
    assert np.all(out[run.labels_preview == 2] == np.array(fallback_color("Sawah"), dtype=np.uint8))
    for c in (3, 4):                                                      # UNNAMED: peta asli run
        sel = run.labels_preview == c
        assert np.array_equal(out[sel], run.map_preview[sel])


def test_recolor_semua_belum_bernama_sama_dengan_map_preview_mock(tmp_path, monkeypatch):
    monkeypatch.delenv("LC_MOCK_SLEEP", raising=False)
    folder = tmp_path / "data" / "lebar"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps({"width": 300, "height": 120, "pixel_size_m": 10}), encoding="utf-8"
    )
    p = RunParams(dataset="lebar", mode="opt", kmeans="omp", k=6, iters=20, seed=42, threads=4)
    run = read_run(MockLauncher(tmp_path / "data", max_side=48).run(p, out_dir_for(tmp_path / "runs", "mock", p)))
    assert validate(run) == []
    assert run.labels_preview.shape[0] != run.labels_preview.shape[1]
    out = recolor(run.labels_preview, class_palette(default_class_map(run.k), run.centroids))
    assert np.array_equal(out, run.map_preview)                          # jalur UI = peta bawaan core


# --------------------------------------------------------------------------- #
# png_bytes
# --------------------------------------------------------------------------- #

def test_png_bytes_lossless_dan_ukuran_benar():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(23, 41, 3), dtype=np.uint8)         # (h, w) tidak persegi
    data = png_bytes(rgb)
    assert isinstance(data, bytes) and data[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "PNG" and im.mode == "RGB" and im.size == (41, 23)
        assert np.array_equal(np.array(im), rgb)


def test_png_bytes_dari_recolor_berukuran_sama_dengan_pratinjau(make_run_dir):
    run = read_run(make_run_dir(width=50, height=21, k=4))
    rgb = recolor(run.labels_preview, class_palette({0: "Vegetasi"}, run.centroids))
    with Image.open(io.BytesIO(png_bytes(rgb))) as im:
        with Image.open(run.path / "labels_preview.png") as ref:
            assert im.size == ref.size == (50, 21)
        assert np.array_equal(np.array(im), rgb)


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 4), dtype=np.uint8),
    np.zeros((4, 4, 3), dtype=np.int64), np.zeros((4, 4, 3), dtype=np.float64),
    np.zeros((0, 4, 3), dtype=np.uint8), np.zeros((4, 0, 3), dtype=np.uint8),
])
def test_png_bytes_bentuk_atau_tipe_salah_ditolak(bad):
    with pytest.raises(ValueError, match="rgb"):
        png_bytes(bad)


# --------------------------------------------------------------------------- #
# table_csv_bytes
# --------------------------------------------------------------------------- #

def parse_csv(data: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))


def test_csv_header_area_table_dan_cluster_table():
    counts = np.array([1, 2, 3])
    at = area_table(counts, {0: "Air", 1: "Vegetasi", 2: "Air"})
    ct = cluster_table(counts, RGB4[:3])
    assert table_csv_bytes(at).split(b"\n")[0] == b"kelas,cluster,piksel,persen,luas_ha"
    assert table_csv_bytes(ct).split(b"\n")[0] == b"cluster,r,g,b,piksel,persen,luas_ha"


def test_csv_isi_area_table_dibulatkan_2_desimal_dan_total_100():
    counts = np.array([1, 2, 3])                  # total 6 piksel @10 m: 0,06 ha
    at = area_table(counts, {0: "Air", 1: "Vegetasi", 2: "Lainnya"})
    data = table_csv_bytes(at)
    assert parse_csv(data) == [
        ["kelas", "cluster", "piksel", "persen", "luas_ha"],
        ["Air", "0", "1", "16.67", "0.01"],
        ["Vegetasi", "1", "2", "33.33", "0.02"],
        ["Lainnya", "2", "3", "50.00", "0.03"],
        ["Total", "", "6", "100.00", "0.06"],
    ]
    assert CSV_DECIMALS == 2


def test_csv_format_byte_utf8_tanpa_bom_akhir_baris_lf():
    at = area_table(np.array([5, 5]), {0: "Air", 1: "Vegetasi"})
    data = table_csv_bytes(at)
    assert not data.startswith(b"\xef\xbb\xbf")                       # tanpa BOM
    assert b"\r" not in data and data.endswith(b"\n")                 # LF di semua platform
    assert data.decode("utf-8").splitlines()[0].startswith("kelas,")  # kolom pertama bersih
    assert len(data.decode("utf-8").splitlines()) == 4                # header + 2 kelas + Total


def test_csv_cluster_digabung_diberi_kutip_dan_baris_total_kosong():
    at = area_table(np.array([10, 20, 30]), {0: "Air", 1: "Vegetasi", 2: "Air"})
    data = table_csv_bytes(at)
    assert b'Air,"0, 2",40,66.67,0.40\n' in data
    assert data.rstrip(b"\n").endswith(b"Total,,60,100.00,0.60")
    rows = parse_csv(data)
    assert rows[1] == ["Air", "0, 2", "40", "66.67", "0.40"] and rows[-1][1] == ""
    assert all(len(r) == 5 for r in rows)


def test_csv_tidak_mengubah_dataframe_masukan():
    at = area_table(np.array([1, 2, 3]), {0: "Air", 1: "Vegetasi", 2: "Lainnya"})
    before = at.copy(deep=True)
    table_csv_bytes(at)
    pd.testing.assert_frame_equal(at, before)
    assert at["persen"].dtype == np.float64 and at["persen"].iloc[0] == 1 * 100.0 / 6   # tidak dibulatkan


def test_csv_tile_penuh_tanpa_notasi_ilmiah():
    counts = np.array([60_280_200, 60_280_200])                       # 120 560 400 piksel
    at = area_table(counts, {0: "Air", 1: "Vegetasi"})
    rows = parse_csv(table_csv_bytes(at))
    assert rows[-1] == ["Total", "", "120560400", "100.00", "1205604.00"]
    assert rows[1] == ["Air", "0", "60280200", "50.00", "602802.00"]


def test_csv_cluster_berjumlah_nol_tampil_nol():
    at = area_table(np.array([64, 0, 0, 0]), {0: "Vegetasi", 1: "Air", 2: "Air", 3: "Lainnya"})
    rows = {r[0]: r for r in parse_csv(table_csv_bytes(at))[1:]}
    assert rows["Air"] == ["Air", "1, 2", "0", "0.00", "0.00"]
    assert rows["Vegetasi"][3] == "100.00" and rows["Total"][2] == "64"
    ct = parse_csv(table_csv_bytes(cluster_table(np.array([64, 0]), RGB4[:2])))
    assert ct[2] == ["1", "201", "198", "190", "0", "0.00", "0.00"]


def test_csv_cluster_table_isi_dan_kolom_integer_tetap_integer():
    ct = cluster_table(np.array([1000, 3000]), RGB4[:2])
    rows = parse_csv(table_csv_bytes(ct))
    assert rows[1] == ["0", "34", "61", "40", "1000", "25.00", "10.00"]
    assert rows[2] == ["1", "201", "198", "190", "3000", "75.00", "30.00"]


def test_csv_nama_kelas_dengan_koma_kutip_dan_non_ascii_utuh():
    name = 'Sawah, "tadah hujan" é'
    at = area_table(np.array([3, 1]), {0: name, 1: "Air"})
    data = table_csv_bytes(at)
    assert data.decode("utf-8")                                        # UTF-8 sah
    assert parse_csv(data)[1][0] == name


def test_csv_decimals_dan_validasi():
    at = area_table(np.array([1, 2]), {0: "Air", 1: "Vegetasi"})
    assert parse_csv(table_csv_bytes(at, decimals=0))[1][3:] == ["33", "0"]
    assert parse_csv(table_csv_bytes(at, decimals=4))[1][3:] == ["33.3333", "0.0100"]
    for bad in (-1, 11, True, 2.0, "2", None):
        with pytest.raises(ValueError, match="decimals"):
            table_csv_bytes(at, decimals=bad)


def test_csv_masukan_salah_ditolak():
    with pytest.raises(ValueError, match="DataFrame"):
        table_csv_bytes([["kelas"], ["Air"]])
    for bad in (float("nan"), float("inf")):
        df = pd.DataFrame({"kelas": ["Air"], "persen": [bad], "luas_ha": [1.0]})
        with pytest.raises(ValueError, match="persen"):
            table_csv_bytes(df)


def test_csv_df_lain_tanpa_kolom_persen_tetap_bisa_diekspor():
    df = pd.DataFrame({"x": [0.1, 2.0], "nama": ["a", "b"]})
    assert table_csv_bytes(df) == b"x,nama\n0.1,a\n2.0,b\n"
    assert table_csv_bytes(df.iloc[0:0]) == b"x,nama\n"                # hanya header


# --------------------------------------------------------------------------- #
# export_full_res_png
# --------------------------------------------------------------------------- #

# Labels tangan, lebar 5 x tinggi 3 (tidak persegi, tidak simetris): salah tukar w/h atau
# salah urutan baris/kolom langsung terlihat di piksel tertentu.
LABELS_5X3 = np.array(
    [[0, 1, 2, 3, 0],
     [1, 1, 1, 2, 2],
     [3, 3, 0, 0, 1]], dtype=np.uint8,
)
PAL4 = np.array([[10, 20, 30], [200, 10, 10], [10, 200, 10], [10, 10, 200]], dtype=np.uint8)


def write_u8(path: Path, labels: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(labels, dtype=np.uint8).tobytes())
    return path


def read_back(path: Path) -> tuple[Image.Image, np.ndarray]:
    with Image.open(path) as im:
        im.load()
        return im.copy(), np.array(im.convert("RGB"))


def test_export_penuh_label_tangan_piksel_per_piksel(tmp_path):
    src = write_u8(tmp_path / "labels.u8", LABELS_5X3)
    out = tmp_path / "penuh.png"
    assert export_full_res_png(src, 5, 3, PAL4, out) is None
    im, rgb = read_back(out)
    assert im.mode == "P" and im.size == (5, 3)                        # (lebar, tinggi)
    assert rgb.shape == (3, 5, 3)
    assert np.array_equal(rgb, PAL4[LABELS_5X3])
    # Piksel tertentu (x = kolom, y = baris): salah tukar w/h atau transpos membuatnya gagal.
    assert tuple(rgb[0, 4]) == tuple(PAL4[0])      # x=4,y=0 -> label 0
    assert tuple(rgb[0, 3]) == tuple(PAL4[3])      # x=3,y=0 -> label 3
    assert tuple(rgb[1, 3]) == tuple(PAL4[2])      # x=3,y=1 -> label 2
    assert tuple(rgb[2, 0]) == tuple(PAL4[3])      # x=0,y=2 -> label 3
    assert tuple(rgb[2, 4]) == tuple(PAL4[1])      # x=4,y=2 -> label 1


def test_export_penuh_palet_tersimpan_di_png_dan_indeks_sama_dengan_label(tmp_path):
    out = tmp_path / "penuh.png"
    export_full_res_png(write_u8(tmp_path / "l.u8", LABELS_5X3), 5, 3, PAL4, out)
    with Image.open(out) as im:
        assert im.getpalette()[:12] == PAL4.reshape(-1).tolist()
        assert np.array_equal(np.array(im), LABELS_5X3)                # indeks palet = id cluster


def test_export_penuh_sama_dengan_recolor_pada_citra_lebih_besar(tmp_path):
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 16, size=(300, 421), dtype=np.uint8)      # tinggi 300, lebar 421
    pal = class_palette({i: "Air" for i in range(0, 16, 3)}, rng.integers(0, 256, size=(16, 3)))
    out = tmp_path / "besar.png"
    export_full_res_png(write_u8(tmp_path / "l.u8", labels), 421, 300, pal, out)
    _, rgb = read_back(out)
    assert np.array_equal(rgb, recolor(labels, pal))                   # jalur penuh = jalur pratinjau


def test_export_penuh_menerima_str_dan_membuat_folder_tujuan(tmp_path):
    src = write_u8(tmp_path / "l.u8", LABELS_5X3)
    out = tmp_path / "belum" / "ada" / "penuh.png"
    export_full_res_png(str(src), 5, 3, PAL4, str(out))
    assert out.is_file()


def test_export_penuh_k_1_dan_k_16(tmp_path):
    one = np.zeros((4, 6), dtype=np.uint8)
    export_full_res_png(write_u8(tmp_path / "a.u8", one), 6, 4, [[1, 2, 3]], tmp_path / "a.png")
    assert np.all(read_back(tmp_path / "a.png")[1] == [1, 2, 3])
    pal16 = np.arange(48, dtype=np.uint8).reshape(16, 3)
    lab16 = (np.arange(16 * 2, dtype=np.uint8) % 16).reshape(2, 16)
    export_full_res_png(write_u8(tmp_path / "b.u8", lab16), 16, 2, pal16, tmp_path / "b.png")
    assert np.array_equal(read_back(tmp_path / "b.png")[1], pal16[lab16])


def assert_tidak_menulis(out: Path):
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


@pytest.mark.parametrize("w, h", [(5, 4), (4, 3), (6, 3), (15, 2), (1, 1)])
def test_export_penuh_ukuran_file_tidak_cocok_ditolak(tmp_path, w, h):
    src = write_u8(tmp_path / "l.u8", LABELS_5X3)                       # 15 byte
    out = tmp_path / "x.png"
    if w * h == 15:                                                    # ukuran sama => lolos (w,h memang ambigu)
        export_full_res_png(src, w, h, PAL4, out)
        assert out.is_file()
        return
    with pytest.raises(ValueError, match=rf"ukuran 15 byte.*{w} \* {h} = {w * h}"):
        export_full_res_png(src, w, h, PAL4, out)
    assert_tidak_menulis(out)


def test_export_penuh_file_terpotong_ditolak(tmp_path):
    src = tmp_path / "l.u8"
    src.write_bytes(LABELS_5X3.tobytes()[:-1])                          # kurang 1 byte
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="14 byte"):
        export_full_res_png(src, 5, 3, PAL4, out)
    assert_tidak_menulis(out)


def test_export_penuh_label_di_luar_palet_ditolak(tmp_path):
    labels = LABELS_5X3.copy()
    labels[1, 2] = 4                                                    # palet hanya 0..3
    src = write_u8(tmp_path / "l.u8", labels)
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match=r"label terbesar 4.*0\.\.3"):
        export_full_res_png(src, 5, 3, PAL4, out)
    assert_tidak_menulis(out)


def test_export_penuh_palet_lebih_dari_256_atau_salah_ditolak(tmp_path):
    src = write_u8(tmp_path / "l.u8", LABELS_5X3)
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="256"):
        export_full_res_png(src, 5, 3, np.zeros((257, 3), dtype=np.uint8), out)
    for bad in (np.zeros((4, 2), dtype=np.uint8), np.full((4, 3), 256, dtype=np.int64), np.zeros((4, 3))):
        with pytest.raises(ValueError, match="palette"):
            export_full_res_png(src, 5, 3, bad, out)
    assert_tidak_menulis(out)


@pytest.mark.parametrize("w, h", [(0, 3), (5, 0), (-5, 3), (5.0, 3), (True, 3), ("5", 3), (None, 3)])
def test_export_penuh_dimensi_salah_ditolak(tmp_path, w, h):
    src = write_u8(tmp_path / "l.u8", LABELS_5X3)
    with pytest.raises(ValueError, match="integer >= 1"):
        export_full_res_png(src, w, h, PAL4, tmp_path / "x.png")


def test_export_penuh_file_tidak_ada(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_full_res_png(tmp_path / "hilang.u8", 5, 3, PAL4, tmp_path / "x.png")
    assert_tidak_menulis(tmp_path / "x.png")


def test_export_penuh_file_kosong_ditolak(tmp_path):
    src = tmp_path / "kosong.u8"
    src.write_bytes(b"")
    with pytest.raises(ValueError, match="0 byte"):
        export_full_res_png(src, 5, 3, PAL4, tmp_path / "x.png")


def test_export_penuh_menimpa_atomik_tanpa_sisa_tmp(tmp_path):
    out = tmp_path / "penuh.png"
    out.write_bytes(b"lama, bukan PNG")
    export_full_res_png(write_u8(tmp_path / "l.u8", LABELS_5X3), 5, 3, PAL4, out)
    assert np.array_equal(read_back(out)[1], PAL4[LABELS_5X3])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["l.u8", "penuh.png"]


def test_export_penuh_gagal_menulis_membersihkan_tmp_dan_menjaga_file_lama(tmp_path, monkeypatch):
    out = tmp_path / "penuh.png"
    out.write_bytes(b"lama")
    src = write_u8(tmp_path / "l.u8", LABELS_5X3)

    def boom(*args, **kwargs):
        raise OSError("disk penuh (tiruan)")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk penuh"):
        export_full_res_png(src, 5, 3, PAL4, out)
    monkeypatch.undo()
    assert out.read_bytes() == b"lama"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["l.u8", "penuh.png"]   # .tmp dibersihkan


def test_export_penuh_dari_run_mock_nyata_tidak_persegi(tmp_path, monkeypatch):
    # labels.u8 buatan tangan dari pratinjau sebuah run nyata: ekspor penuh = pewarnaan ulang pratinjau.
    monkeypatch.delenv("LC_MOCK_SLEEP", raising=False)
    folder = tmp_path / "data" / "lebar"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"width": 300, "height": 120}), encoding="utf-8")
    p = RunParams(dataset="lebar", mode="cpu", kmeans="auto", k=5, iters=10, seed=1, threads=1)
    run = read_run(MockLauncher(tmp_path / "data", max_side=60).run(p, out_dir_for(tmp_path / "runs", "mock", p)))
    h, w = run.labels_preview.shape
    assert w != h
    pal = class_palette({0: "Air", 2: "Air", 3: "Sawah"}, run.centroids)
    out = tmp_path / "peta_penuh.png"
    export_full_res_png(write_u8(tmp_path / "labels.u8", run.labels_preview), w, h, pal, out)
    assert np.array_equal(read_back(out)[1], recolor(run.labels_preview, pal))
