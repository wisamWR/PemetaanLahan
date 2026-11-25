"""Tes API halaman Pemetaan (Fase 2 UI web): kartu citra, unggah, gambar run, luas, ekspor.

Memakai klien ASGI kecil (`asgi_client.py`), folder sementara, dan folder run buatan `write_run_dir`
(K=4, 64x48 piksel; `default_labels`: tiap cluster 768 piksel = 25 %). Tanpa jaringan dan tanpa httpx.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

import asgi_client as http
from api.server import create_app
from api.settings import Settings
from conftest import default_labels, write_run_dir
from landcover.contract import read_run
from landcover.labeling import CLASS_PRESETS, UNNAMED
from launchers.mock import MockLauncher
from services import datasets as svc_datasets
from services.labeling_panel import area_csv, class_map_from_names, cluster_csv

RID = "foto__opt-auto__k4_it20_s42_t8"
W, H, K = 64, 48, 4
PNG = {"Content-Type": "image/png"}


def png_bytes(w: int = 16, h: int = 12, color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings("mock", tmp_path / "data", tmp_path / "runs", "tidak-dipakai", None)


@pytest.fixture
def app(settings):
    def make(backend, data_dir, core_bin):
        return MockLauncher(data_dir, max_side=48)

    return create_app(settings, launcher_factory=make)


@pytest.fixture
def run(settings):
    """Run K=4 di runs/mock/<RID>, dataset `foto` (PNG 64x48 di data/foto.png)."""
    (settings.data_dir).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)).save(settings.data_dir / "foto.png")
    write_run_dir(settings.runs_dir / "mock" / RID, width=W, height=H, k=K, meta={"dataset": "foto"})
    return RID


def names_q(names) -> dict:
    return {"names": json.dumps(names)}


def decode(res: http.Response) -> Image.Image:
    return Image.open(io.BytesIO(res.body))


# --- kartu citra dan pratinjau -------------------------------------------------------------------- #

def test_datasets_memuat_kartu_dengan_ukuran_dan_batas_unggah(app, settings):
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "foto.png").write_bytes(png_bytes(40, 30))
    folder = settings.data_dir / "tile_a"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"width": 10980, "height": 10980}), encoding="utf-8")

    data = http.get(app, "/api/datasets").json()
    cards = {d["name"]: d for d in data["datasets"]}
    assert (cards["foto"]["width"], cards["foto"]["height"], cards["foto"]["kind"]) == (40, 30, "png")
    assert cards["foto"]["size_text"] == "40 × 30 piksel" and cards["foto"]["has_preview"] is True
    assert cards["foto"]["mock"] is False
    assert cards["tile_a"]["size_text"] == "10,980 × 10,980 piksel"
    assert cards["tile_a"]["has_preview"] is False          # folder tanpa preview.jpg
    assert cards["tiruan_sleman"]["mock"] is True and cards["tiruan_sleman"]["has_preview"] is True
    assert cards["tiruan_sleman"]["size_text"] == "10,980 × 10,980 piksel"
    assert data["upload"]["max_bytes"] == svc_datasets.MAX_UPLOAD_BYTES
    assert "PNG" in data["upload"]["hint"] and "25 MB" in data["upload"]["hint"]


def test_preview_sintetis_png_dan_folder(app, settings):
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "foto.png").write_bytes(png_bytes(40, 30, (200, 10, 10)))
    folder = settings.data_dir / "tile_a"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"width": 20, "height": 10}), encoding="utf-8")
    Image.new("RGB", (20, 10), (0, 0, 255)).save(folder / "preview.jpg")

    sint = http.get(app, "/api/datasets/tiruan_sleman/preview", size=128)
    assert sint.status == 200 and sint.headers["content-type"] == "image/jpeg"
    assert max(decode(sint).size) == 128 and decode(sint).size[0] == decode(sint).size[1]   # tile persegi

    png = http.get(app, "/api/datasets/foto/preview")
    assert png.status == 200 and decode(png).size == (40, 30)          # tidak diperbesar
    r, g, b = decode(png).convert("RGB").getpixel((5, 5))
    assert r > 150 and g < 60 and b < 60

    tile = http.get(app, "/api/datasets/tile_a/preview")
    assert tile.status == 200 and decode(tile).size == (20, 10)


def test_preview_etag_memberi_304(app):
    first = http.get(app, "/api/datasets/tiruan_sleman/preview", size=64)
    etag = first.headers["etag"]
    again = http.request(app, "GET", "/api/datasets/tiruan_sleman/preview", params={"size": 64}, headers={"If-None-Match": etag})
    assert again.status == 304 and again.body == b""


def test_preview_folder_tanpa_preview_jpg_404_dengan_alasan(app, settings):
    folder = settings.data_dir / "tile_a"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"width": 20, "height": 10}), encoding="utf-8")
    res = http.get(app, "/api/datasets/tile_a/preview")
    assert res.status == 404 and "preview.jpg" in res.json()["detail"]


@pytest.mark.parametrize("name", ["tidak_ada", "..", "%2E%2E", "a%2Fb", "..%2F..%2Fsecret", "foto.png"])
def test_preview_menolak_nama_yang_bukan_dataset(app, name):
    res = http.get(app, f"/api/datasets/{name}/preview")
    assert res.status == 404


def test_preview_ukuran_di_luar_batas_422(app):
    assert http.get(app, "/api/datasets/tiruan_sleman/preview", size=5).status == 422
    assert http.get(app, "/api/datasets/tiruan_sleman/preview", size=99999).status == 422


# --- unggah ----------------------------------------------------------------------------------------- #

def upload(app, name: str, data: bytes):
    return http.post(app, "/api/datasets", params={"filename": name}, body=data, content_type="image/png")


def test_unggah_png_menjadi_dataset_baru_lengkap_dengan_pratinjau(app, settings):
    res = upload(app, "Citra Baru.png", png_bytes(24, 16))
    assert res.status == 201, res.text
    card = res.json()
    assert card["name"] == "Citra Baru" and card["kind"] == "png" and (card["width"], card["height"]) == (24, 16)
    assert card["has_preview"] is True and card["mock"] is False
    assert (settings.data_dir / "uploads" / "Citra Baru.png").is_file()
    listing = http.get(app, "/api/datasets").json()["datasets"]
    assert "Citra Baru" in [d["name"] for d in listing]
    preview = http.get(app, f"/api/datasets/{quote('Citra Baru')}/preview")
    assert preview.status == 200 and decode(preview).size == (24, 16)


def test_unggah_nama_sama_ditolak_409_dan_berkas_lama_utuh(app, settings):
    assert upload(app, "a.png", png_bytes(8, 6, (1, 1, 1))).status == 201
    res = upload(app, "a.png", png_bytes(8, 6, (2, 2, 2)))
    assert res.status == 409 and "sudah ada" in res.json()["detail"]
    assert (settings.data_dir / "uploads" / "a.png").read_bytes() == png_bytes(8, 6, (1, 1, 1))


@pytest.mark.parametrize("payload", [b"", b"bukan gambar", b"\x89PNG\r\n\x1a\n" + b"rusak"])
def test_unggah_berkas_rusak_422(app, settings, payload):
    res = upload(app, "x.png", payload)
    assert res.status == 422
    assert not (settings.data_dir / "uploads").exists() or not list((settings.data_dir / "uploads").iterdir())


def test_unggah_jpeg_berkedok_png_422(app):
    buf = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buf, format="JPEG")
    res = upload(app, "x.png", buf.getvalue())
    assert res.status == 422 and "bukan PNG" in res.json()["detail"]


@pytest.mark.parametrize("filename", ["con.png", "..", "///.png"])
def test_unggah_nama_tidak_aman_422(app, filename):
    assert upload(app, filename, png_bytes()).status == 422


def test_unggah_tanpa_filename_422(app):
    assert http.post(app, "/api/datasets", body=png_bytes(), content_type="image/png").status == 422


def test_unggah_melewati_batas_byte_413(app, settings, monkeypatch):
    monkeypatch.setattr("api.server.MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(svc_datasets, "MAX_UPLOAD_BYTES", 100)
    data = png_bytes(64, 64)                        # > 100 byte
    assert len(data) > 100
    res = upload(app, "besar.png", data)
    assert res.status == 413 and "MB" in res.json()["detail"]
    assert not (settings.data_dir / "uploads").exists()


def test_unggah_melewati_batas_piksel_413(app, settings, monkeypatch):
    monkeypatch.setattr(svc_datasets, "MAX_UPLOAD_PIXELS", 100)
    res = upload(app, "lebar.png", png_bytes(20, 20))
    assert res.status == 413 and "piksel" in res.json()["detail"]
    assert not (settings.data_dir / "uploads").exists()


# --- gambar run ------------------------------------------------------------------------------------------ #

def test_labels_dan_map_preview_apa_adanya(app, run, settings):
    folder = settings.runs_dir / "mock" / run
    for route, name in (("labels_preview", "labels_preview.png"), ("map_preview", "map_preview.png")):
        res = http.get(app, f"/api/runs/{run}/{route}")
        assert res.status == 200 and res.headers["content-type"] == "image/png"
        assert res.body == (folder / name).read_bytes()
    assert np.array_equal(np.asarray(decode(http.get(app, f"/api/runs/{run}/labels_preview"))), default_labels(W, H, K))


@pytest.mark.parametrize("route", ["labels_preview", "map_preview", "map", "original"])
def test_gambar_run_tak_dikenal_404(app, run, route):
    assert http.get(app, f"/api/runs/tidak_ada/{route}").status == 404
    assert http.get(app, f"/api/runs/..%2F..%2Fmeta/{route}").status == 404
    assert http.get(app, f"/api/runs/%2E%2E/{route}").status == 404


def test_original_dari_png_dataset_dan_sintetis(app, run, settings):
    res = http.get(app, f"/api/runs/{run}/original")
    assert res.status == 200 and res.headers["content-type"] == "image/jpeg"
    assert decode(res).size == (W, H)
    assert decode(http.get(app, f"/api/runs/{run}/original", size=32)).size == (32, 24)

    sid = "tiruan_sleman__opt-auto__k4_it20_s42_t8"
    write_run_dir(settings.runs_dir / "mock" / sid, width=W, height=H, k=K,
                  meta={"dataset": "tiruan_sleman", "source_image": "synthetic"})
    assert decode(http.get(app, f"/api/runs/{sid}/original")).size == (W, H)


def test_original_tidak_ada_404_dengan_alasan(app, settings):
    rid = "hilang__opt-auto__k4_it20_s42_t8"
    write_run_dir(settings.runs_dir / "mock" / rid, width=W, height=H, k=K, meta={"dataset": "hilang"})
    res = http.get(app, f"/api/runs/{rid}/original")
    assert res.status == 404 and "Citra asli tidak ditemukan" in res.json()["detail"]


def test_map_berwarna_kelas_dari_names(app, run):
    names = ["Vegetasi", "Air", UNNAMED, ""]
    res = http.get(app, f"/api/runs/{run}/map", **names_q(names))
    assert res.status == 200 and res.headers["content-type"] == "image/png"
    img = np.asarray(decode(res).convert("RGB"))
    assert img.shape == (H, W, 3)
    labels = default_labels(W, H, K)
    assert (img[labels == 0] == (34, 139, 34)).all()                 # Vegetasi
    assert (img[labels == 1] == (30, 100, 200)).all()                # Air
    centroids = read_run(Path(next(iter(_run_dirs(app))))).centroids
    assert (img[labels == 2] == centroids[2]).all()                  # tanpa nama = warna centroid
    assert (img[labels == 3] == centroids[3]).all()


def _run_dirs(app):
    root = app.state.settings.runs_dir / "mock"
    return [str(p) for p in root.iterdir() if (p / "meta.json").is_file()]


def test_map_default_dan_centroid_sama_dengan_map_preview(app, run, settings):
    expected = np.asarray(Image.open(settings.runs_dir / "mock" / run / "map_preview.png").convert("RGB"))
    for params in ({}, {"view": "centroid"}, {"view": "centroid", **names_q(["Air"] * K)}):
        got = np.asarray(decode(http.get(app, f"/api/runs/{run}/map", **params)).convert("RGB"))
        assert np.array_equal(got, expected), params


def test_map_etag_dan_cap_run_tidak_mengubah_gambar(app, run):
    first = http.get(app, f"/api/runs/{run}/map", v="123", **names_q(["Air"] * K))
    again = http.request(app, "GET", f"/api/runs/{run}/map", params={"v": "999", **names_q(["Air"] * K)},
                         headers={"If-None-Match": first.headers["etag"]})
    assert again.status == 304


@pytest.mark.parametrize("params", [
    {"names": "bukan json"}, {"names": '{"0": "Air"}'}, {"names": json.dumps(["Air"] * (K - 1))},
    {"names": json.dumps(["Air"] * (K + 1))}, {"names": json.dumps([1, 2, 3, 4])},
    {"names": json.dumps(["Total", "", "", ""])}, {"names": json.dumps(["x" * 41, "", "", ""])},
    {"view": "lain"},
])
def test_map_masukan_salah_422(app, run, params):
    assert http.get(app, f"/api/runs/{run}/map", **params).status == 422


# --- luas dan pelabelan ------------------------------------------------------------------------------------ #

def area(app, rid, class_map=None):
    res = http.post(app, f"/api/runs/{rid}/area", json={"class_map": class_map or {}})
    return res


def test_area_tanpa_nama_semua_belum_diberi_nama(app, run):
    res = area(app, run)
    assert res.status == 200, res.text
    d = res.json()
    assert d["run_id"] == run and d["k"] == K and d["mock"] is True
    assert d["mock_notice"] == "DATA TIRUAN: bukan hasil pengukuran"
    assert d["names"] == [UNNAMED] * K
    assert d["presets"] == list(CLASS_PRESETS) and d["unnamed_label"] == UNNAMED and d["total_label"] == "Total"
    assert d["preview"] == {"width": W, "height": H} and d["name_max_chars"] == 40
    assert len(d["fingerprint"]) == 16 and d["errors"] == []
    # kalimat bantuan dan keterangan datang dari Python (sama dengan editor Streamlit)
    assert d["help"]["panel"].startswith("Beri nama tiap cluster: pilih Kelas atau ketik Nama lain.")
    assert "mengalahkan kolom Kelas" in d["help"]["other"]
    assert "'Total' tidak boleh dipakai" in d["help"]["other"] and "berwarna sama di peta" in d["help"]["kelas"]
    assert d["preview_caption"] == (
        "Pratinjau 64 × 48 piksel dari citra 64 × 48. "
        "Persen dan luas dihitung dari seluruh piksel (counts), bukan dari pratinjau."
    )
    assert d["area_caption"] == "Persen dari seluruh piksel (counts); luas dihitung dengan 10 m per piksel."
    assert d["export_note"] == "Berkas unduhan hasil tiruan berawalan TIRUAN_."
    assert [e["kind"] for e in d["exports"]] == ["map", "area", "clusters", "full"]
    assert [c["cluster"] for c in d["clusters"]] == [0, 1, 2, 3]
    assert all(c["pixels"] == 768 and c["percent_text"] == "25.00 %" and not c["named"] for c in d["clusters"])
    # satu baris "Belum diberi nama" (semua cluster) + Total
    assert [(r["name"], r["clusters"], r["percent_text"]) for r in d["area"]] == [
        (UNNAMED, "0, 1, 2, 3", "100.00 %"), ("Total", "", "100.00 %"),
    ]
    assert d["area"][0]["unnamed"] is True and len(d["area"][0]["colors"]) == K       # warna centroid tiap cluster
    assert d["area"][1]["is_total"] is True and d["area"][1]["area_text"] == "30.72 ha"   # 3072 piksel x 10 m
    assert [k["key"] for k in d["kpis"]] == ["total", "named", "largest", "unnamed"]
    assert d["kpis"][0]["value"] == "30.72 ha" and d["kpis"][2]["value"] == "Belum ada"
    assert d["kpis"][3]["value"] == "100.00 %"


def test_area_kunci_teks_dan_dua_bentuk_nilai(app, run):
    d = area(app, run, {"0": "Vegetasi", "1": {"kelas": "Air"}, "2": {"nama_lain": "Sawah"}}).json()
    assert d["names"] == ["Vegetasi", "Air", "Sawah", UNNAMED]
    assert [(c["kelas"], c["nama_lain"]) for c in d["clusters"]] == [("Vegetasi", ""), ("Air", ""), (None, "Sawah"), (None, "")]
    assert d["palette"][0] == "#228b22" and d["palette"][1] == "#1e64c8"
    assert d["kpis"][1]["value"] == "3" and d["kpis"][2]["value"].endswith("(25.0 %)")


def test_area_nama_lain_mengalahkan_kelas(app, run):
    d = area(app, run, {"0": {"kelas": "Air", "nama_lain": "Sawah"}, "1": {"kelas": "Air", "nama_lain": "   "}}).json()
    assert d["names"][:2] == ["Sawah", "Air"]               # spasi saja = kosong, jadi Kelas dipakai
    c0 = d["clusters"][0]
    assert (c0["name"], c0["kelas"], c0["nama_lain"]) == ("Sawah", None, "Sawah")
    # warna Sawah = warna cadangan deterministik dari nama, bukan Air
    assert d["palette"][0] != d["palette"][1]


def test_area_cluster_berkelas_sama_digabung_dan_berwarna_sama(app, run):
    d = area(app, run, {"0": "Vegetasi", "2": {"kelas": "Vegetasi"}}).json()
    veg = next(r for r in d["area"] if r["name"] == "Vegetasi")
    assert veg["clusters"] == "0, 2" and veg["cluster_ids"] == [0, 2] and veg["pixels"] == 1536
    assert veg["percent_text"] == "50.00 %" and veg["area_text"] == "15.36 ha" and veg["colors"] == ["#228b22"]
    assert d["palette"][0] == d["palette"][2] == "#228b22"
    assert [r["name"] for r in d["area"]] == ["Vegetasi", UNNAMED, "Total"]
    assert d["kpis"][1]["value"] == "1"                     # satu kelas berbeda
    assert d["kpis"][3]["value"] == "50.00 %"


def test_area_nama_total_ditolak_dan_dilaporkan_tanpa_500(app, run):
    d = area(app, run, {"0": "Total", "1": {"nama_lain": " total "}, "2": "Air"}).json()
    assert d["names"] == [UNNAMED, UNNAMED, "Air", UNNAMED]
    assert len(d["errors"]) == 2 and all("Total" in e and "dipertahankan" in e for e in d["errors"])
    assert [c["invalid"] for c in d["clusters"]] == [True, True, False, False]


@pytest.mark.parametrize("body", [
    {"class_map": {"4": "Air"}},                            # K = 4: id 0..3
    {"class_map": {"-1": "Air"}},
    {"class_map": {"x": "Air"}},
    {"class_map": {"０": "Air"}},                           # angka bukan ASCII
    {"class_map": {"1.5": "Air"}},
    {"class_map": {"0": 3}},
    {"class_map": {"0": {"kelas": 3}}},
    {"class_map": {"0": {"lain": "x"}}},
    {"class_map": {"0": "x" * 41}},
    {"class_map": {"0": {"kelas": "Air", "nama_lain": "y" * 41}}},
    {"class_map": [1, 2]},
    {"peta": {}},
])
def test_area_masukan_salah_422(app, run, body):
    res = http.post(app, f"/api/runs/{run}/area", json=body)
    assert res.status == 422 and isinstance(res.json()["detail"], str)


def test_area_tanpa_badan_json_dianggap_kosong(app, run):
    res = http.post(app, f"/api/runs/{run}/area", json={})
    assert res.status == 200 and res.json()["names"] == [UNNAMED] * K


def test_area_run_tidak_ada_404(app, run):
    assert area(app, "tidak_ada").status == 404
    assert area(app, "..").status == 404


def test_area_tanpa_keadaan_dan_sidik_stabil(app, run, settings):
    body = {"0": "Air", "1": {"nama_lain": "Sawah"}}
    first, second = area(app, run, body).json(), area(app, run, body).json()
    assert first == second
    write_run_dir(settings.runs_dir / "mock" / RID, width=W, height=H, k=K, meta={"dataset": "foto"})   # ditimpa, hasil sama
    assert area(app, run, body).json()["fingerprint"] == first["fingerprint"]
    write_run_dir(settings.runs_dir / "mock" / RID, width=W, height=H, k=K, meta={"dataset": "foto"},
                  counts=[1000, 1, 1, 1070])
    assert area(app, run, body).json()["fingerprint"] != first["fingerprint"]


def test_area_hasil_janggal_422_bukan_500(app, settings):
    rid = "janggal__opt-auto__k4_it20_s42_t8"
    write_run_dir(settings.runs_dir / "mock" / rid, width=W, height=H, k=K, centroids=[[300, 0, 0]] * K)
    res = area(app, rid)
    assert res.status == 422 and "0..255" in res.json()["detail"]


def test_area_cluster_berjumlah_nol_tampil_nol_persen(app, settings):
    rid = "kosong__opt-auto__k4_it20_s42_t8"
    labels = np.zeros((H, W), dtype=np.int64)
    labels[:, 32:] = 1                                       # cluster 2 dan 3 kosong
    write_run_dir(settings.runs_dir / "mock" / rid, width=W, height=H, k=K, labels=labels)
    d = area(app, rid, {"2": "Air"}).json()
    row = next(r for r in d["area"] if r["name"] == "Air")
    assert row["pixels"] == 0 and row["percent_text"] == "0.00 %" and row["area_text"] == "0.00 ha"


# --- ekspor ------------------------------------------------------------------------------------------------- #

def test_ekspor_csv_sama_dengan_fungsi_services_dan_berawalan_tiruan(app, run, settings):
    names = ["Vegetasi", "Air", "Sawah", UNNAMED]
    loaded = read_run(settings.runs_dir / "mock" / run)
    class_map = class_map_from_names(names, K)

    res = http.get(app, f"/api/runs/{run}/export/area", **names_q(names))
    assert res.status == 200 and res.headers["content-type"].startswith("text/csv")
    assert res.body == area_csv(loaded, class_map)
    disposition = res.headers["content-disposition"]
    assert disposition.startswith("attachment;") and f"TIRUAN_luas-per-kelas_{run}.csv" in disposition

    res = http.get(app, f"/api/runs/{run}/export/clusters")
    assert res.body == cluster_csv(loaded) and f"TIRUAN_per-cluster_{run}.csv" in res.headers["content-disposition"]


def test_ekspor_peta_png_memakai_warna_kelas(app, run):
    res = http.get(app, f"/api/runs/{run}/export/map", **names_q(["Air", "", "", ""]))
    assert res.status == 200 and res.headers["content-type"] == "image/png"
    assert f"TIRUAN_peta-kelas_{run}.png" in res.headers["content-disposition"]
    img = np.asarray(decode(res).convert("RGB"))
    assert (img[default_labels(W, H, K) == 0] == (30, 100, 200)).all()
    assert http.get(app, f"/api/runs/{run}/export/map").body == http.get(app, f"/api/runs/{run}/map").body


def test_area_hasil_nyata_tanpa_catatan_tiruan_dan_dengan_ukuran_piksel_lain(app, settings):
    rid = "nyata2__opt-auto__k4_it20_s42_t8"
    write_run_dir(settings.runs_dir / "mock" / rid, width=W, height=H, k=K,
                  meta={"mock": False, "dataset": "nyata2", "pixel_size_m": None})       # tanpa pixel_size_m: bawaan 10 m + catatan
    d = area(app, rid).json()
    assert d["mock"] is False and d["mock_notice"] is None and d["export_note"] is None
    assert d["pixel_size_note"] and "pixel_size_m" in d["pixel_size_note"]
    assert d["area_caption"].endswith(d["pixel_size_note"]) and d["area_caption"].startswith("Persen dari seluruh piksel")
    assert all(not e["filename"].startswith("TIRUAN_") for e in d["exports"])


def test_ekspor_hasil_nyata_tanpa_awalan_tiruan(app, settings):
    rid = "nyata__opt-auto__k4_it20_s42_t8"
    write_run_dir(settings.runs_dir / "mock" / rid, width=W, height=H, k=K, meta={"mock": False, "dataset": "nyata"})
    res = http.get(app, f"/api/runs/{rid}/export/area")
    assert f'filename="luas-per-kelas_{rid}.csv"' in res.headers["content-disposition"]


def test_ekspor_nama_berkas_non_ascii_memakai_filename_bintang(app, settings):
    rid = "pétá__opt-auto__k4_it20_s42_t8"
    write_run_dir(settings.runs_dir / "mock" / rid, width=W, height=H, k=K, meta={"dataset": "pétá"})
    res = http.get(app, f"/api/runs/{quote(rid)}/export/area")
    assert res.status == 200
    assert "filename*=UTF-8''TIRUAN_luas-per-kelas_p%C3%A9t%C3%A1__opt-auto" in res.headers["content-disposition"]
    assert 'filename="TIRUAN_luas-per-kelas_p_t___opt-auto' in res.headers["content-disposition"]


def test_ekspor_png_penuh_tidak_ada_tanpa_labels_u8(app, run):
    res = http.get(app, f"/api/runs/{run}/export/full")
    assert res.status == 404 and "labels.u8" in res.json()["detail"]
    full = next(e for e in area(app, run).json()["exports"] if e["kind"] == "full")
    assert full["available"] is False and "labels.u8" in full["reason"]


def test_ekspor_png_penuh_dari_labels_u8_dan_bersih_sesudahnya(app, run, settings, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    labels = default_labels(W, H, K).astype(np.uint8)
    (settings.runs_dir / "mock" / run / "labels.u8").write_bytes(labels.tobytes())

    exports = {e["kind"]: e for e in area(app, run).json()["exports"]}
    assert exports["full"]["available"] is True and exports["full"]["reason"] is None
    assert exports["full"]["filename"] == f"TIRUAN_peta-kelas-penuh_{run}.png"

    res = http.get(app, f"/api/runs/{run}/export/full", **names_q(["Air", "", "", ""]))
    assert res.status == 200 and res.headers["content-type"] == "image/png"
    assert f"TIRUAN_peta-kelas-penuh_{run}.png" in res.headers["content-disposition"]
    img = decode(res)
    assert img.size == (W, H) and img.mode == "P"
    assert np.array_equal(np.asarray(img), labels)                       # indeks piksel = id cluster
    assert tuple(img.getpalette()[:3]) == (30, 100, 200)                 # warna Air untuk cluster 0
    assert list(scratch.iterdir()) == []                                  # berkas sementara dihapus
    assert not list((settings.runs_dir / "mock" / run).glob("*.png.tmp"))


def test_ekspor_png_penuh_labels_u8_salah_ukuran_422_dan_bersih(app, run, settings, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    (settings.runs_dir / "mock" / run / "labels.u8").write_bytes(b"\x00" * 10)
    res = http.get(app, f"/api/runs/{run}/export/full")
    assert res.status == 422 and "ekspor gagal" in res.json()["detail"]
    assert list(scratch.iterdir()) == []


def test_ekspor_masukan_salah(app, run):
    assert http.get(app, f"/api/runs/{run}/export/lain").status == 404
    assert http.get(app, "/api/runs/tidak_ada/export/area").status == 404
    assert http.get(app, f"/api/runs/{run}/export/area", names="bukan json").status == 422
    assert http.get(app, f"/api/runs/{run}/export/area", **names_q(["Air"])).status == 422     # K salah
    assert http.get(app, f"/api/runs/{run}/export/map", **names_q(["Total", "", "", ""])).status == 422


# --- alur utuh: Buat peta -> luas -> ekspor ------------------------------------------------------------------ #

def test_buat_peta_optimalisasi_lalu_luas_dan_ekspor(app):
    body = {"dataset": "tiruan_sleman", "mode": "opt", "k": 4, "iters": 3, "seed": 7}
    res = http.post(app, "/api/runs", json=body)
    assert res.status == 202, res.text
    job = res.json()
    assert http.wait_job(app, job["job_id"])["status"] == "done"
    [rid] = job["run_ids"]

    # citra asli sintetis sama ukurannya dengan peta pratinjau; luas dari counts resolusi penuh (tile 10980x10980)
    original, labels = (decode(http.get(app, f"/api/runs/{rid}/{r}")) for r in ("original", "labels_preview"))
    assert original.size == labels.size
    d = area(app, rid, {"0": "Air", "1": {"kelas": "Vegetasi"}}).json()
    assert d["kpis"][0]["value"] == "1,205,604.00 ha" and d["mock"] is True
    assert sum(c["pixels"] for c in d["clusters"]) == 10980 * 10980
    assert http.get(app, f"/api/runs/{rid}/export/area", **names_q(d["names"])).status == 200
    full = http.get(app, f"/api/runs/{rid}/export/full")                       # run tiruan: tanpa labels.u8
    assert full.status == 404


def test_area_dan_ekspor_mengecualikan_nodata_murni_dan_campuran(app, settings):
    # Kasus 1: cluster nodata murni (cluster 0 seluruhnya nodata)
    rid_pure = "ndmurni__opt-auto__k4_it20_s42_t8"
    write_run_dir(
        settings.runs_dir / "mock" / rid_pure, width=W, height=H, k=K,
        counts=[300, 772, 1000, 1000],
        meta={"mock": False, "dataset": "ndmurni", "nodata_pixels": 300, "nodata_cluster": 0},
    )
    dp = area(app, rid_pure, {"1": "Vegetasi", "2": "Air"}).json()
    assert dp["nodata"]["pixels"] == 300 and dp["nodata"]["cluster"] == 0
    assert "300 piksel nodata" in dp["nodata"]["note"] and "dikecualikan dari luas" in dp["nodata"]["note"]
    assert dp["clusters"][0]["all_nodata"] is True and dp["clusters"][0]["nodata_badge"] == "Seluruhnya nodata"
    assert dp["clusters"][0]["pixels"] == 0 and dp["clusters"][1]["all_nodata"] is False
    assert [r["name"] for r in dp["area"]] == ["Vegetasi", "Air", UNNAMED, "Total", "Nodata (dikecualikan)"]
    total_row, nd_row = dp["area"][3], dp["area"][4]
    assert total_row["is_total"] is True and total_row["pixels"] == 2772 and total_row["percent_text"] == "100.00 %"
    assert nd_row["is_nodata"] is True and nd_row["pixels"] == 300 and nd_row["clusters"] == "0"
    assert total_row["area_ha"] + nd_row["area_ha"] == pytest.approx(W * H * 0.01)

    csv_text = http.get(app, f"/api/runs/{rid_pure}/export/area", **names_q(dp["names"])).body.decode("utf-8")
    assert "Nodata (dikecualikan),0,300," in csv_text

    # Kasus 2: nodata bercampur dengan piksel valid di cluster yang sama (cluster 0 punya 300 nodata + 200 valid)
    rid_mix = "ndcampur__opt-auto__k4_it20_s42_t8"
    write_run_dir(
        settings.runs_dir / "mock" / rid_mix, width=W, height=H, k=K,
        counts=[500, 572, 1000, 1000],
        meta={"mock": False, "dataset": "ndcampur", "nodata_pixels": 300, "nodata_cluster": 0},
    )
    dm = area(app, rid_mix, {"0": "Air"}).json()
    assert dm["clusters"][0]["all_nodata"] is False
    assert dm["clusters"][0]["pixels"] == 200 and dm["clusters"][0]["nodata_pixels"] == 300
    assert "300" in dm["clusters"][0]["nodata_badge"]
    assert [r["name"] for r in dm["area"]] == ["Air", UNNAMED, "Total", "Nodata (dikecualikan)"]
    assert next(r for r in dm["area"] if r["name"] == "Air")["pixels"] == 200
