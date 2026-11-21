"""Tes `landcover.labeling` (plan Langkah 4): luas, persen, penggabungan kelas, cluster kosong."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from landcover.contract import read_run, validate
from landcover.labeling import (
    CLASS_PRESETS,
    TOTAL_LABEL,
    UNNAMED,
    area_table,
    cluster_table,
    default_class_map,
    normalize_class_name,
)
from launchers.base import RunParams, out_dir_for
from launchers.mock import MockLauncher

TILE = 10980
TILE_PIXELS = TILE * TILE          # 120 560 400 (angka dari plan)
TILE_HA = 1_205_604                # 12 056,04 km2

RGB4 = [[34, 61, 40], [201, 198, 190], [60, 120, 200], [180, 140, 60]]


def rows(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Baris tabel luas menurut nama kelas."""
    return {r["kelas"]: r for _, r in df.iterrows()}


# --------------------------------------------------------------------------- #
# default_class_map dan konstanta
# --------------------------------------------------------------------------- #

def test_konstanta_sesuai_plan():
    assert CLASS_PRESETS == ["Vegetasi", "Air", "Lahan terbangun", "Lahan terbuka", "Awan/bayangan", "Lainnya"]
    assert UNNAMED == "Belum diberi nama"
    assert TOTAL_LABEL == "Total"
    assert TOTAL_LABEL not in CLASS_PRESETS and UNNAMED not in CLASS_PRESETS


def test_default_class_map_semua_cluster_belum_bernama():
    assert default_class_map(3) == {0: UNNAMED, 1: UNNAMED, 2: UNNAMED}
    assert default_class_map(np.int64(2)) == {0: UNNAMED, 1: UNNAMED}


@pytest.mark.parametrize("k", [0, -1, 2.5, "8", None])
def test_default_class_map_k_salah_ditolak(k):
    with pytest.raises(ValueError, match="k harus"):
        default_class_map(k)


def test_default_class_map_objek_baru_tiap_panggilan():
    a = default_class_map(2)
    a[0] = "Air"
    assert default_class_map(2)[0] == UNNAMED


# --------------------------------------------------------------------------- #
# cluster_table
# --------------------------------------------------------------------------- #

def test_cluster_table_kolom_urutan_dan_tipe():
    df = cluster_table([100, 300, 0, 600], RGB4)
    assert list(df.columns) == ["cluster", "r", "g", "b", "piksel", "persen", "luas_ha"]
    assert df["cluster"].tolist() == [0, 1, 2, 3]
    assert df[["r", "g", "b"]].values.tolist() == RGB4
    assert df["piksel"].tolist() == [100, 300, 0, 600]
    assert df["piksel"].dtype == np.int64
    assert df["persen"].dtype == np.float64 and df["luas_ha"].dtype == np.float64
    assert df["persen"].tolist() == [10.0, 30.0, 0.0, 60.0]


def test_seribu_piksel_pada_10_m_sama_dengan_10_ha():
    df = cluster_table([1000], [[1, 2, 3]], pixel_size_m=10.0)
    assert df["luas_ha"].tolist() == [10.0]
    assert df["persen"].tolist() == [100.0]
    # parameter bawaan = 10 m, dan bilangan bulat diterima sebagai ukuran piksel
    assert cluster_table([1000], [[1, 2, 3]])["luas_ha"].tolist() == [10.0]
    assert cluster_table([1000], [[1, 2, 3]], pixel_size_m=10)["luas_ha"].tolist() == [10.0]


def test_ukuran_piksel_lain_dipakai_kuadrat():
    # ortofoto drone 0,5 m: 1 piksel = 0,25 m2 -> 40 000 piksel = 1 ha
    df = cluster_table([40_000, 80_000], [[0, 0, 0], [9, 9, 9]], pixel_size_m=0.5)
    assert df["luas_ha"].tolist() == pytest.approx([1.0, 2.0])
    # 20 m: satu piksel = 400 m2 = 0,04 ha
    assert cluster_table([50], [[0, 0, 0]], pixel_size_m=20)["luas_ha"].tolist() == pytest.approx([2.0])


def test_satu_tile_penuh_eksak():
    counts = [TILE_PIXELS // 2, TILE_PIXELS // 4, TILE_PIXELS - TILE_PIXELS // 2 - TILE_PIXELS // 4]
    df = cluster_table(counts, [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert df["piksel"].sum() == TILE_PIXELS == 120_560_400
    assert df["luas_ha"].sum() == TILE_HA == 1_205_604
    assert df["luas_ha"].sum() / 100 == pytest.approx(12_056.04)       # km2
    assert df["persen"].sum() == pytest.approx(100.0, abs=1e-9)
    assert df["luas_ha"].tolist() == pytest.approx([c * 0.01 for c in counts], rel=1e-12)


def test_persen_tidak_dibulatkan_di_modul():
    df = cluster_table([1, 2], [[0, 0, 0], [1, 1, 1]])
    assert df["persen"].iloc[0] == pytest.approx(100 / 3, rel=1e-15)
    assert abs(df["persen"].iloc[0] - round(df["persen"].iloc[0], 2)) > 1e-4


def test_persen_dihitung_dari_integer_tanpa_error_pembulatan():
    # 3 dari 10^8 piksel: dihitung piksel*100/total, jadi eksak sampai ujung double
    df = cluster_table([3, 10**8 - 3], [[0, 0, 0], [1, 1, 1]])
    assert df["persen"].iloc[0] == 3 * 100.0 / 10**8


def test_cluster_berjumlah_nol_aman_di_cluster_table():
    df = cluster_table([64, 0, 0, 0], RGB4)
    assert df["piksel"].tolist() == [64, 0, 0, 0]
    assert df["persen"].tolist() == [100.0, 0.0, 0.0, 0.0]
    assert df["luas_ha"].tolist() == [0.64, 0.0, 0.0, 0.0]
    assert not df.isna().any().any()


def test_semua_counts_nol_tidak_menghasilkan_nan_atau_error():
    df = cluster_table([0, 0], [[0, 0, 0], [1, 1, 1]])
    assert df["persen"].tolist() == [0.0, 0.0] and df["luas_ha"].tolist() == [0.0, 0.0]
    t = area_table([0, 0], {0: "Air"})
    assert not t.isna().any().any()
    assert t["persen"].tolist() == [0.0, 0.0, 0.0] and t["piksel"].tolist() == [0, 0, 0]


def test_cluster_table_menerima_array_run_result_dan_tidak_mengubah_masukan():
    counts = np.array([5, 15], dtype=np.int64)
    centroids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    before = (counts.copy(), centroids.copy())
    cluster_table(counts, centroids)
    assert (counts == before[0]).all() and (centroids == before[1]).all()
    # dtype integer lain (mis. uint8 untuk centroid, int32 untuk counts) juga diterima
    df = cluster_table(counts.astype(np.int32), centroids.astype(np.uint8))
    assert df["piksel"].dtype == np.int64 and df["r"].tolist() == [1, 4]


# --------------------------------------------------------------------------- #
# area_table
# --------------------------------------------------------------------------- #

def test_area_table_kolom_dan_baris_total_terakhir():
    t = area_table([100, 300, 600], {0: "Air", 1: "Vegetasi", 2: "Lahan terbuka"})
    assert list(t.columns) == ["kelas", "cluster", "piksel", "persen", "luas_ha"]
    assert t["kelas"].tolist() == ["Air", "Vegetasi", "Lahan terbuka", "Total"]
    assert t["cluster"].tolist() == ["0", "1", "2", ""]
    assert t["piksel"].tolist() == [100, 300, 600, 1000]
    assert t["persen"].tolist() == [10.0, 30.0, 60.0, 100.0]
    assert t["luas_ha"].tolist() == [1.0, 3.0, 6.0, 10.0]
    assert t["piksel"].dtype == np.int64


def test_cluster_dengan_nama_sama_digabung():
    t = area_table([100, 200, 300, 400], {0: "Vegetasi", 1: "Air", 2: "Vegetasi", 3: "Lahan terbuka"})
    r = rows(t)
    assert t["kelas"].tolist() == ["Vegetasi", "Air", "Lahan terbuka", "Total"]   # urut id cluster terkecil
    assert r["Vegetasi"]["piksel"] == 100 + 300 and r["Vegetasi"]["cluster"] == "0, 2"
    assert r["Vegetasi"]["persen"] == 40.0 and r["Vegetasi"]["luas_ha"] == 4.0
    assert r["Air"]["piksel"] == 200 and r["Lahan terbuka"]["piksel"] == 400


def test_gabungan_dua_cluster_sama_dengan_jumlah_keduanya_dari_cluster_table():
    counts = [123, 4567, 89, 1011, 12]
    per_cluster = cluster_table(counts, [[i, i, i] for i in range(5)])
    t = area_table(counts, {0: "Air", 1: "Vegetasi", 2: "Air", 3: "Vegetasi", 4: "Lainnya"})
    r = rows(t)
    for kelas, ids in {"Air": [0, 2], "Vegetasi": [1, 3], "Lainnya": [4]}.items():
        sub = per_cluster[per_cluster["cluster"].isin(ids)]
        assert r[kelas]["piksel"] == sub["piksel"].sum()
        assert r[kelas]["persen"] == pytest.approx(sub["persen"].sum(), rel=1e-12)
        assert r[kelas]["luas_ha"] == pytest.approx(sub["luas_ha"].sum(), rel=1e-12)


@pytest.mark.parametrize("w,h,k", [(64, 48, 4), (257, 131, 8), (10, 10, 16), (1, 1, 1)])
def test_total_persen_100_dan_total_piksel_sama_dengan_w_kali_h(w, h, k):
    n = w * h
    base = n // k
    counts = [base + (1 if i < n - base * k else 0) for i in range(k)]
    assert sum(counts) == n
    names = {i: CLASS_PRESETS[i % 3] for i in range(k)}
    t = area_table(counts, names)
    body, total = t.iloc[:-1], t.iloc[-1]
    assert total["kelas"] == "Total" and total["piksel"] == w * h
    assert total["persen"] == 100.0 and body["persen"].sum() == pytest.approx(100.0, abs=0.01)
    assert body["piksel"].sum() == w * h
    assert total["luas_ha"] == pytest.approx(w * h * 0.01)


@pytest.mark.parametrize("n", [3, 6, 7, 11])
def test_persen_total_tepat_100_walau_jumlah_persen_kelas_meleset(n):
    # 6, 7, dan 11 kelas berisi satu piksel: jumlah persen kelas = 100.00000000000001 di float
    t = area_table([1] * n, {i: f"Kelas {i}" for i in range(n)})
    assert t.iloc[-1]["persen"] == 100.0
    assert t["persen"].iloc[:-1].sum() == pytest.approx(100.0, abs=1e-9)


def test_satu_tile_penuh_di_area_table():
    counts = [TILE_PIXELS // 2, TILE_PIXELS - TILE_PIXELS // 2]
    t = area_table(counts, {0: "Vegetasi", 1: "Air"})
    total = t.iloc[-1]
    assert total["piksel"] == 120_560_400 and total["luas_ha"] == 1_205_604
    assert total["luas_ha"] / 100 == pytest.approx(12_056.04)
    assert t["persen"].tolist() == [50.0, 50.0, 100.0]


def test_cluster_berjumlah_nol_tanpa_exception_di_area_table():
    # kontrak U2: citra satu warna, K = 4
    counts = [64, 0, 0, 0]
    t = area_table(counts, {0: "Vegetasi", 1: "Air", 2: "Air", 3: "Lahan terbuka"})
    r = rows(t)
    assert r["Air"]["piksel"] == 0 and r["Air"]["persen"] == 0.0 and r["Air"]["luas_ha"] == 0.0
    assert r["Air"]["cluster"] == "1, 2"                       # kelas kosong tetap tampil
    assert r["Lahan terbuka"]["piksel"] == 0
    assert r["Vegetasi"]["persen"] == 100.0 and r["Total"]["persen"] == 100.0
    assert not t.isna().any().any()


def test_kelas_semua_cluster_nol_tidak_membagi_dengan_piksel_kelas():
    # satu kelas berisi dua cluster kosong, kelas lain berisi semua piksel
    t = area_table([50, 0, 0], {0: "Vegetasi", 1: "Air", 2: "Air"})
    assert rows(t)["Air"]["persen"] == 0.0 and np.isfinite(t["persen"]).all()


def test_cluster_tanpa_nama_digabung_jadi_satu_baris_belum_diberi_nama():
    t = area_table([10, 20, 30, 40], {1: "Air"})                # 0, 2, 3 tidak ada di peta
    assert t["kelas"].tolist() == [UNNAMED, "Air", "Total"]
    r = rows(t)
    assert r[UNNAMED]["cluster"] == "0, 2, 3" and r[UNNAMED]["piksel"] == 10 + 30 + 40
    assert r["Air"]["piksel"] == 20


def test_peta_kelas_bawaan_semua_belum_bernama():
    t = area_table([30, 70], default_class_map(2))
    assert t["kelas"].tolist() == [UNNAMED, "Total"]
    assert t["cluster"].tolist() == ["0, 1", ""] and t["piksel"].tolist() == [100, 100]


def test_nama_dirapikan_sebelum_digabung():
    t = area_table([1, 2, 3], {0: "Air", 1: "  Air  ", 2: "   "})
    assert t["kelas"].tolist() == ["Air", UNNAMED, "Total"]
    assert rows(t)["Air"]["piksel"] == 3 and rows(t)["Air"]["cluster"] == "0, 1"


def test_nama_berbeda_huruf_besar_kecil_tidak_digabung():
    # penggabungan memakai nama persis (setelah dirapikan); UI memakai SelectboxColumn
    t = area_table([1, 2], {0: "Air", 1: "air"})
    assert t["kelas"].tolist() == ["Air", "air", "Total"]


def test_kunci_peta_kelas_numpy_int_diterima_dan_peta_tidak_diubah():
    cm = {np.int64(0): "Air", np.int32(1): "Vegetasi"}
    snapshot = dict(cm)
    t = area_table([1, 2], cm)
    assert t["kelas"].tolist() == ["Air", "Vegetasi", "Total"]
    assert cm == snapshot


def test_area_table_tidak_mengubah_counts():
    counts = np.array([3, 4], dtype=np.int64)
    area_table(counts, {0: "Air", 1: "Air"})
    assert counts.tolist() == [3, 4]


# --------------------------------------------------------------------------- #
# Masukan salah
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "counts,pesan",
    [
        ([], "tidak kosong"),
        ([[1, 2], [3, 4]], "1 dimensi"),
        ([1, -1, 5], "negatif"),
        ([1.0, 2.0], "integer"),
        ([True, False], "integer"),
        (["1", "2"], "integer"),
    ],
)
def test_counts_salah_ditolak(counts, pesan):
    with pytest.raises(ValueError, match=pesan):
        area_table(counts, {})
    n = len(counts)
    with pytest.raises(ValueError, match=pesan):
        cluster_table(counts, np.zeros((n, 3), dtype=np.int64))


@pytest.mark.parametrize(
    "centroids,pesan",
    [
        ([[1, 2], [3, 4]], r"\(K, 3\)"),
        ([[1, 2, 3]], "tidak sama"),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "tidak sama"),
        ([1, 2], r"\(K, 3\)"),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "integer"),
    ],
)
def test_centroids_salah_ditolak(centroids, pesan):
    with pytest.raises(ValueError, match=pesan):
        cluster_table([1, 2], centroids)


@pytest.mark.parametrize("size", [0, -10, 0.0, float("nan"), float("inf"), "10", None, True])
def test_pixel_size_salah_ditolak(size):
    with pytest.raises(ValueError, match="pixel_size_m"):
        cluster_table([1], [[0, 0, 0]], pixel_size_m=size)
    with pytest.raises(ValueError, match="pixel_size_m"):
        area_table([1], {0: "Air"}, pixel_size_m=size)


@pytest.mark.parametrize("key", [2, -1, 99])
def test_id_cluster_di_luar_rentang_ditolak(key):
    with pytest.raises(ValueError, match="di luar rentang 0..1"):
        area_table([1, 2], {key: "Air"})


@pytest.mark.parametrize("key", ["0", 0.5, None])
def test_kunci_peta_kelas_bukan_integer_ditolak(key):
    with pytest.raises(ValueError, match="id cluster harus integer"):
        area_table([1, 2], {key: "Air"})


@pytest.mark.parametrize("nama", [5, 3.5, ["Air"], b"Air", float("inf")])
def test_nama_kelas_bukan_teks_ditolak(nama):
    with pytest.raises(ValueError, match="harus teks"):
        area_table([1, 2], {0: nama})


@pytest.mark.parametrize("sel_kosong", [None, float("nan"), np.nan, pd.NA, np.float64("nan")])
def test_sel_kosong_dari_data_editor_berarti_belum_diberi_nama(sel_kosong):
    # st.data_editor mengembalikan None / NaN / pd.NA bila SelectboxColumn dikosongkan
    assert normalize_class_name(sel_kosong) == UNNAMED
    t = area_table([1, 2, 3], {0: sel_kosong, 1: "Air", 2: ""})
    assert t["kelas"].tolist() == [UNNAMED, "Air", "Total"]
    assert rows(t)[UNNAMED]["cluster"] == "0, 2" and rows(t)[UNNAMED]["piksel"] == 4


@pytest.mark.parametrize("nama", ["Total", "total", "  TOTAL ", "ToTaL"])
def test_nama_total_dicadangkan_untuk_baris_jumlah(nama):
    with pytest.raises(ValueError, match="dicadangkan"):
        area_table([1, 2], {0: nama})
    with pytest.raises(ValueError, match="dicadangkan"):
        normalize_class_name(nama)


def test_normalize_class_name():
    assert normalize_class_name("  Hutan bakau ") == "Hutan bakau"
    assert normalize_class_name("") == UNNAMED and normalize_class_name("\t \n") == UNNAMED
    assert normalize_class_name("Total tanah") == "Total tanah"       # hanya nama persis "Total" ditolak
    with pytest.raises(ValueError):
        normalize_class_name(3)


# --------------------------------------------------------------------------- #
# Dengan keluaran MockLauncher dan read_run (jalur yang dipakai UI)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _tanpa_tidur(monkeypatch):
    monkeypatch.delenv("LC_MOCK_SLEEP", raising=False)


def mock_run(tmp_path, dataset: str, k: int):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    launcher = MockLauncher(data_dir, max_side=48)
    p = RunParams(dataset=dataset, mode="opt", kmeans="cuda", k=k, iters=20, seed=42, threads=8)
    run = read_run(launcher.run(p, out_dir_for(tmp_path / "runs", "mock", p)))
    assert validate(run) == []
    return run


def test_run_mock_citra_satu_warna_cluster_kosong(tmp_path):
    folder = tmp_path / "data" / "seragam"
    folder.mkdir(parents=True)
    Image.fromarray(np.full((8, 8, 3), (30, 90, 40), dtype=np.uint8)).save(folder / "tile.png")
    run = mock_run(tmp_path, "seragam", k=4)
    assert run.counts.tolist() == [64, 0, 0, 0]

    size = run.meta["pixel_size_m"]
    ct = cluster_table(run.counts, run.centroids, size)
    assert ct["persen"].tolist() == [100.0, 0.0, 0.0, 0.0] and ct["luas_ha"].tolist() == [0.64, 0.0, 0.0, 0.0]
    at = area_table(run.counts, {0: "Vegetasi", 1: "Air", 2: "Air", 3: "Lainnya"}, size)
    assert rows(at)["Air"]["persen"] == 0.0 and at.iloc[-1]["piksel"] == run.n_pixels == 64
    assert not at.isna().any().any() and not ct.isna().any().any()


def test_run_mock_tile_penuh_memakai_counts_resolusi_penuh(tmp_path):
    # manifest 10980 x 10980 -> counts diskalakan ke resolusi penuh (bukan ke ukuran pratinjau)
    folder = tmp_path / "data" / "tile"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps({"width": TILE, "height": TILE, "pixel_size_m": 10}), encoding="utf-8"
    )
    run = mock_run(tmp_path, "tile", k=8)
    assert run.n_pixels == TILE_PIXELS and int(run.counts.sum()) == 120_560_400

    size = run.meta["pixel_size_m"]
    at = area_table(run.counts, {i: CLASS_PRESETS[i % 4] for i in range(8)}, size)
    total = at.iloc[-1]
    assert total["piksel"] == 120_560_400 and total["luas_ha"] == 1_205_604
    assert at["persen"].iloc[:-1].sum() == pytest.approx(100.0, abs=0.01)
    ct = cluster_table(run.counts, run.centroids, size)
    assert len(ct) == 8 and ct["piksel"].tolist() == run.counts.tolist()
    assert ct["luas_ha"].sum() == pytest.approx(1_205_604, rel=1e-12)
