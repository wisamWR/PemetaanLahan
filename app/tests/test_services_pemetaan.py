"""Logika murni halaman Pemetaan (Fase 2 UI web) di `services/`, tanpa API dan tanpa browser."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from conftest import write_run_dir
from landcover.contract import read_run
from landcover.datasets import KIND_PNG, KIND_SYNTHETIC, Dataset, DatasetExistsError
from landcover.labeling import UNNAMED
from services import datasets as sd
from services.labeling_panel import (
    NAME_MAX_CHARS, centroid_hex, class_map_from_names, class_map_from_rows, cluster_error_ids,
    export_items, hex_color, names_of, palette_hex, run_fingerprint,
)
from services.maps import jpeg_bytes, thumbnail


def png(w=16, h=12, color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


# --- gambar ----------------------------------------------------------------------------------------- #

def test_thumbnail_mengecilkan_tanpa_memperbesar_dan_menjaga_rasio():
    img = np.zeros((400, 1000, 3), dtype=np.uint8)
    assert thumbnail(img, 250).shape == (100, 250, 3)
    assert thumbnail(img, 5000) is img or thumbnail(img, 5000).shape == img.shape
    assert thumbnail(np.zeros((1, 9, 3), dtype=np.uint8), 3).shape == (1, 3, 3)       # minimal satu piksel


@pytest.mark.parametrize("bad", [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.float32),
                                 np.zeros((0, 4, 3), dtype=np.uint8), np.zeros((4, 4, 4), dtype=np.uint8)])
def test_thumbnail_dan_jpeg_menolak_bentuk_salah(bad):
    with pytest.raises(ValueError):
        thumbnail(bad, 10)
    with pytest.raises(ValueError):
        jpeg_bytes(bad)


def test_jpeg_bytes_menghasilkan_jpeg_yang_bisa_dibaca():
    img = np.full((20, 30, 3), 128, dtype=np.uint8)
    out = Image.open(io.BytesIO(jpeg_bytes(img)))
    assert out.format == "JPEG" and out.size == (30, 20)


# --- peta kelas dari isian -------------------------------------------------------------------------------- #

def test_class_map_dari_isian_nama_lain_mengalahkan_kelas():
    cm, errors = class_map_from_rows({0: ("Air", "Sawah"), 1: ("Air", None), 2: (None, "  Kebun  "), 3: ("Air", "  ")}, 5)
    assert errors == []
    assert cm == {0: "Sawah", 1: "Air", 2: "Kebun", 3: "Air", 4: UNNAMED}


def test_class_map_dari_isian_menolak_total_dan_kembali_unnamed():
    cm, errors = class_map_from_rows({0: ("Air", "Total"), 1: ("Total", None), 2: (None, "TOTAL")}, 3)
    assert list(cm.values()) == [UNNAMED] * 3
    assert cluster_error_ids(errors) == {0, 1, 2}                 # juga "Total" lewat kolom Kelas


def test_class_map_dari_isian_menolak_id_dan_panjang():
    with pytest.raises(ValueError, match="di luar rentang"):
        class_map_from_rows({3: (None, "x")}, 3)
    with pytest.raises(ValueError, match="di luar rentang"):
        class_map_from_rows({-1: (None, "x")}, 3)
    with pytest.raises(ValueError, match="terlalu panjang"):
        class_map_from_rows({0: (None, "x" * (NAME_MAX_CHARS + 1))}, 3)
    class_map_from_rows({0: (None, "x" * NAME_MAX_CHARS)}, 3)           # tepat di batas: boleh


def test_class_map_dari_names():
    assert class_map_from_names(None, 3) == {0: UNNAMED, 1: UNNAMED, 2: UNNAMED}
    assert class_map_from_names(["Air", "", UNNAMED], 3) == {0: "Air", 1: UNNAMED, 2: UNNAMED}
    assert names_of({0: "Air", 2: "Sawah"}, 3) == ["Air", UNNAMED, "Sawah"]
    for bad in (["Air"], ["a", "b", "c", "d"], ["a", 1, "c"], ["Total", "", ""], ["x" * 41, "", ""]):
        with pytest.raises(ValueError):
            class_map_from_names(bad, 3)


def test_hex_dan_palet():
    assert hex_color((0, 10, 255)) == "#000aff" and hex_color((-5, 300, 7)) == "#00ff07"
    run = read_run(write_run_dir(__import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "r", k=3))
    assert centroid_hex(run) == [hex_color(c) for c in run.centroids]
    assert palette_hex(run, {0: "Vegetasi", 1: UNNAMED, 2: "Air"}) == ["#228b22", hex_color(run.centroids[1]), "#1e64c8"]


def test_cluster_error_ids_hanya_pesan_berformat_cluster():
    assert cluster_error_ids(["Cluster 3: x", "Cluster 12: y", "lain", "Cluster : z"]) == {3, 12}


# --- sidik dan ekspor ------------------------------------------------------------------------------------- #

def test_run_fingerprint_stabil_dan_peka_pada_counts_dan_centroid(tmp_path):
    a = read_run(write_run_dir(tmp_path / "a", k=4))
    b = read_run(write_run_dir(tmp_path / "b", k=4))
    c = read_run(write_run_dir(tmp_path / "c", k=4, counts=[1, 2, 3, 4 + 3062]))
    d = read_run(write_run_dir(tmp_path / "d", k=4, centroids=[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]))
    assert run_fingerprint(a) == run_fingerprint(b) and len(run_fingerprint(a)) == 16
    assert len({run_fingerprint(a), run_fingerprint(c), run_fingerprint(d)}) == 3


def test_export_items_nama_berkas_dan_ketersediaan_png_penuh(tmp_path):
    run = read_run(write_run_dir(tmp_path / "r1", k=3))
    items = {i["kind"]: i for i in export_items(run)}
    assert list(items) == ["map", "area", "clusters", "full"]
    assert items["area"]["filename"] == "TIRUAN_luas-per-kelas_r1.csv" and items["area"]["media_type"] == "text/csv"
    assert items["full"]["available"] is False and "labels.u8" in items["full"]["reason"] and items["full"]["hint"] is None
    (tmp_path / "r1" / "labels.u8").write_bytes(b"\x00" * (64 * 48))
    full = {i["kind"]: i for i in export_items(read_run(tmp_path / "r1"))}["full"]
    assert full["available"] is True and full["reason"] is None and "64 × 48" in full["hint"]
    nyata = read_run(write_run_dir(tmp_path / "r2", k=3, meta={"mock": False}))
    assert {i["kind"]: i for i in export_items(nyata)}["map"]["filename"] == "peta-kelas_r2.png"


# --- kartu citra dan unggahan ---------------------------------------------------------------------------- #

def test_dataset_size_dan_cards(tmp_path):
    (tmp_path / "foto.png").write_bytes(png(40, 30))
    folder = tmp_path / "tile"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"width": 7, "height": 5}), encoding="utf-8")
    broken = tmp_path / "rusak"
    broken.mkdir()
    (broken / "manifest.json").write_text("{bukan json", encoding="utf-8")
    cards, skipped = sd.dataset_cards(tmp_path, "cli")
    by_name = {c["name"]: c for c in cards}
    assert by_name["foto"]["size_text"] == "40 × 30 piksel" and by_name["tile"]["size_text"] == "7 × 5 piksel"
    assert by_name["rusak"]["size_text"] == "ukuran tidak diketahui" and by_name["rusak"]["width"] is None
    assert "tiruan_sleman" not in by_name and skipped == ()                     # sintetis hanya pada backend mock
    assert "tiruan_sleman" in {c["name"] for c in sd.dataset_cards(tmp_path, "mock")[0]}


def test_dataset_preview_menolak_citra_terlalu_besar(tmp_path, monkeypatch):
    (tmp_path / "foto.png").write_bytes(png(40, 30))
    monkeypatch.setattr("services.maps.ORIGINAL_MAX_PIXELS", 100)
    monkeypatch.setattr(sd, "ORIGINAL_MAX_PIXELS", 100)
    ds = Dataset("foto", KIND_PNG, tmp_path / "foto.png")
    assert sd.has_preview(tmp_path, ds) is False
    with pytest.raises(sd.PreviewUnavailableError, match="terlalu besar"):
        sd.dataset_preview_array(tmp_path, ds)


def test_dataset_preview_sintetis_sama_dengan_citra_asli_mockrun():
    from launchers.mock import PREVIEW_MAX_SIDE, synthetic_image

    ds = Dataset("tiruan_sleman", KIND_SYNTHETIC, None)
    got = sd.dataset_preview_array("tidak-dipakai", ds)
    assert got.shape == (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE, 3)
    assert np.array_equal(got, synthetic_image("tiruan_sleman", PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))


def test_dataset_preview_jpeg_cache_ikut_waktu_ubah_berkas(tmp_path):
    path = tmp_path / "foto.png"
    path.write_bytes(png(40, 30, (250, 0, 0)))
    ds = Dataset("foto", KIND_PNG, path)
    first, tag1 = sd.dataset_preview_jpeg(tmp_path, ds, 40)
    path.write_bytes(png(40, 30, (0, 0, 250)))                    # diganti dengan isi lain
    import os
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 5_000_000_000))
    second, tag2 = sd.dataset_preview_jpeg(tmp_path, ds, 40)
    assert tag1 != tag2 and first != second


def test_store_upload_batas_dan_kesalahan(tmp_path, monkeypatch):
    ds = sd.store_upload(tmp_path, "baru.png", png(20, 10))
    assert ds.name == "baru" and (tmp_path / "uploads" / "baru.png").is_file()
    with pytest.raises(DatasetExistsError):
        sd.store_upload(tmp_path, "baru.png", png(20, 10))
    with pytest.raises(ValueError, match="kosong"):
        sd.store_upload(tmp_path, "kosong.png", b"")
    with pytest.raises(ValueError):
        sd.store_upload(tmp_path, "x.png", b"bukan gambar")
    monkeypatch.setattr(sd, "MAX_UPLOAD_PIXELS", 100)
    with pytest.raises(sd.UploadTooLargeError, match="piksel"):
        sd.store_upload(tmp_path, "lebar.png", png(20, 20))
    monkeypatch.setattr(sd, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(sd.UploadTooLargeError, match="MB"):
        sd.store_upload(tmp_path, "besar.png", png(20, 10))
    assert sorted(p.name for p in (tmp_path / "uploads").iterdir()) == ["baru.png"]


def test_upload_limits_teks():
    limits = sd.upload_limits()
    assert limits["max_bytes"] == 25 * 1024 * 1024 and limits["max_pixels"] == 16_000_000
    assert limits["hint"].startswith("PNG, maksimal 25 MB dan 16 juta piksel")
