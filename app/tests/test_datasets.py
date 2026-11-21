"""Penemuan dataset, unggahan, dan dukungan PNG lepas di MockLauncher (Langkah 6)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from landcover.datasets import (
    KIND_FOLDER, KIND_PNG, KIND_SYNTHETIC, SYNTHETIC_NAMES, DatasetExistsError,
    list_datasets, resolve_dataset, sanitize_upload_name, save_upload, scan_datasets,
)
from landcover.contract import read_run, validate
from launchers.base import RunParams, is_safe_dataset_name, out_dir_for
from launchers.mock import MockLauncher


def png_bytes(w: int = 8, h: int = 6, color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def write_png(path, w=8, h=6, color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(w, h, color))
    return path


def make_folder_dataset(root, name, with_manifest=True):
    folder = root / name
    folder.mkdir(parents=True)
    if with_manifest:
        (folder / "manifest.json").write_text(json.dumps({"width": 4, "height": 4}), encoding="utf-8")
    return folder


# --- scan ------------------------------------------------------------------ #

def test_folder_data_tidak_ada_berarti_tanpa_dataset(tmp_path):
    scan = scan_datasets(tmp_path / "tidak_ada")
    assert scan.datasets == () and scan.skipped == ()


def test_sumber_dataset_dan_urutan(tmp_path):
    make_folder_dataset(tmp_path, "sleman")
    make_folder_dataset(tmp_path, "tanpa_manifest", with_manifest=False)   # bukan dataset
    write_png(tmp_path / "kecil.png")
    write_png(tmp_path / "unggahan" / "x.png")           # subfolder acak: bukan dataset
    write_png(tmp_path / "uploads" / "dari_ui.png")
    (tmp_path / "catatan.txt").write_text("bukan dataset")

    scan = scan_datasets(tmp_path)
    assert scan.names == ("sleman", "kecil", "dari_ui")
    assert [d.kind for d in scan.datasets] == [KIND_FOLDER, KIND_PNG, KIND_PNG]
    assert scan.get("sleman").path == tmp_path / "sleman"
    assert scan.get("dari_ui").path == tmp_path / "uploads" / "dari_ui.png"
    assert scan.skipped == ()


def test_ekstensi_png_tak_membedakan_huruf_besar(tmp_path):
    write_png(tmp_path / "FOTO.PNG")
    assert scan_datasets(tmp_path).names == ("FOTO",)


def test_nama_tidak_aman_dilewati_dengan_alasan(tmp_path):
    write_png(tmp_path / "foto (1).png")
    write_png(tmp_path / "ok.png")
    scan = scan_datasets(tmp_path)
    assert scan.names == ("ok",)
    assert len(scan.skipped) == 1 and "foto (1).png" in scan.skipped[0] and "tidak aman" in scan.skipped[0]


def test_nama_kembar_prioritas_folder_lalu_png_lalu_unggahan(tmp_path):
    make_folder_dataset(tmp_path, "sama")
    write_png(tmp_path / "sama.png")
    write_png(tmp_path / "uploads" / "sama.png")
    write_png(tmp_path / "uploads" / "lain.png")
    write_png(tmp_path / "lain.png")
    scan = scan_datasets(tmp_path)
    assert scan.get("sama").kind == KIND_FOLDER
    assert scan.get("lain").path == tmp_path / "lain.png"
    # yang kalah: sama.png, uploads/sama.png, uploads/lain.png
    assert len(scan.skipped) == 3 and all("sudah dipakai" in s for s in scan.skipped)
    assert {s.split(":")[0] for s in scan.skipped} == {"sama.png", "uploads/sama.png", "uploads/lain.png"}


def test_resolve_dataset(tmp_path):
    write_png(tmp_path / "kecil.png")
    assert resolve_dataset(tmp_path, "kecil").kind == KIND_PNG
    assert resolve_dataset(tmp_path, "tidak_ada") is None
    assert resolve_dataset(tmp_path, SYNTHETIC_NAMES[0]) is None    # sintetis bukan berkas


def test_dataset_sintetis_hanya_di_backend_mock(tmp_path):
    write_png(tmp_path / "kecil.png")
    mock = list_datasets(tmp_path, "mock")
    assert mock.names == ("kecil", *SYNTHETIC_NAMES)
    assert all(d.kind == KIND_SYNTHETIC and d.path is None for d in mock.datasets[1:])
    assert "tiruan" in mock.datasets[1].label
    assert list_datasets(tmp_path, "cli").names == ("kecil",)
    assert list_datasets(tmp_path / "kosong", "cli").names == ()
    assert list_datasets(tmp_path / "kosong", "mock").names == SYNTHETIC_NAMES


def test_dataset_nyata_menggeser_sintetis_bernama_sama(tmp_path):
    write_png(tmp_path / f"{SYNTHETIC_NAMES[0]}.png")
    scan = list_datasets(tmp_path, "mock")
    assert scan.names.count(SYNTHETIC_NAMES[0]) == 1
    assert scan.get(SYNTHETIC_NAMES[0]).kind == KIND_PNG


# --- nama dan unggahan ----------------------------------------------------- #

@pytest.mark.parametrize("filename, expected", [
    ("peta.png", "peta"),
    ("Peta Sleman 2026.PNG", "Peta Sleman 2026"),
    ("foto (1).png", "foto _1"),
    ("a/b\\c.png", "c"),                       # tanpa komponen path
    ("../../etc/passwd.png", "passwd"),
    ("  .hidden.png", "hidden"),
    ("tile-10m_v2.png", "tile-10m_v2"),
])
def test_sanitize_upload_name(filename, expected):
    name = sanitize_upload_name(filename)
    assert name == expected
    assert is_safe_dataset_name(name)


@pytest.mark.parametrize("filename", ["", "???.png", "___.png", "CON.png", "nul.png", "Lpt1.png"])
def test_sanitize_upload_name_menolak(filename):
    with pytest.raises(ValueError):
        sanitize_upload_name(filename)


def test_save_upload_menulis_ke_uploads_dan_terdeteksi_scan(tmp_path):
    data = png_bytes()
    ds = save_upload(tmp_path, "Peta Baru.png", data)
    assert ds.name == "Peta Baru" and ds.kind == KIND_PNG
    assert ds.path == tmp_path / "uploads" / "Peta Baru.png"
    assert ds.path.read_bytes() == data
    assert scan_datasets(tmp_path).get("Peta Baru") is not None
    assert not list((tmp_path / "uploads").glob("*.tmp"))


def test_save_upload_tidak_menimpa(tmp_path):
    save_upload(tmp_path, "a.png", png_bytes(color=(1, 1, 1)))
    with pytest.raises(DatasetExistsError):
        save_upload(tmp_path, "a.png", png_bytes(color=(2, 2, 2)))
    assert (tmp_path / "uploads" / "a.png").read_bytes() == png_bytes(color=(1, 1, 1))
    write_png(tmp_path / "b.png")                         # bentrok dengan dataset di data/ juga
    with pytest.raises(DatasetExistsError):
        save_upload(tmp_path, "b.png", png_bytes())


@pytest.mark.parametrize("payload", [b"", b"bukan gambar", b"\x89PNG\r\n\x1a\n" + b"rusak"])
def test_save_upload_menolak_berkas_rusak(tmp_path, payload):
    with pytest.raises(ValueError, match="PNG"):
        save_upload(tmp_path, "x.png", payload)
    assert not (tmp_path / "uploads").exists() or not list((tmp_path / "uploads").iterdir())


def test_save_upload_menolak_jpeg_berkedok_png(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="JPEG")
    with pytest.raises(ValueError, match="JPEG"):
        save_upload(tmp_path, "x.png", buf.getvalue())


def test_save_upload_menolak_nama_tidak_bisa_dipakai(tmp_path):
    with pytest.raises(ValueError):
        save_upload(tmp_path, "con.png", png_bytes())


# --- MockLauncher memakai PNG lepas ---------------------------------------- #

def test_mock_memakai_png_lepas_dan_unggahan(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    for rel in ("kecil.png", "uploads/dari_ui.png"):
        (tmp_path / "data" / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(tmp_path / "data" / rel)
    launcher = MockLauncher(tmp_path / "data")

    for name, source in (("kecil", "kecil.png"), ("dari_ui", "dari_ui.png")):
        p = RunParams(name, "cpu", "auto", 4, 3, 42, 1)
        out = launcher.run(p, out_dir_for(tmp_path / "runs", "mock", p))
        run = read_run(out)
        assert validate(run) == []
        assert (run.width, run.height) == (40, 30)          # ukuran citra, bukan DEFAULT_SIZE
        assert run.meta["source_image"] == source
        assert run.counts.sum() == 40 * 30


def test_mock_nama_tanpa_berkas_tetap_sintetis(tmp_path):
    p = RunParams("tidak_ada", "cpu", "auto", 3, 2, 42, 1)
    out = MockLauncher(tmp_path / "data").run(p, out_dir_for(tmp_path / "runs", "mock", p))
    assert read_run(out).meta["source_image"] == "synthetic"
