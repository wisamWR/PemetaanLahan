"""Uji pembaca kontrak: folder valid diterima, folder rusak ditolak dengan pesan jelas."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from conftest import write_run_dir
from landcover.contract import (
    RUN_FILES,
    STAGES,
    BatchResult,
    ContractError,
    RunResult,
    StageTiming,
    read_any,
    read_batch,
    read_run,
    validate,
    validate_batch,
)


def has(problems: list[str], *fragments: str) -> bool:
    """Ada satu pesan masalah yang memuat semua potongan teks."""
    return any(all(f in p for f in fragments) for p in problems)


# --------------------------------------------------------------------------- #
# Folder valid
# --------------------------------------------------------------------------- #

def test_folder_valid_dibaca_dan_lolos_validate(make_run_dir):
    run = read_run(make_run_dir())
    assert isinstance(run, RunResult)
    assert validate(run) == []
    assert (run.k, run.width, run.height, run.n_pixels) == (4, 64, 48, 64 * 48)
    assert run.mock is True
    assert (run.mode, run.kmeans_device) == ("opt", "cuda")
    assert tuple(run.stages) == STAGES


def test_tipe_dan_bentuk_array(make_run_dir):
    run = read_run(make_run_dir())
    assert run.centroids.dtype == np.int64 and run.centroids.shape == (4, 3)
    assert run.counts.dtype == np.int64 and run.counts.shape == (4,)
    assert run.labels_preview.dtype == np.uint8 and run.labels_preview.shape == (48, 64)
    assert run.map_preview.dtype == np.uint8 and run.map_preview.shape == (48, 64, 3)
    assert int(run.counts.sum()) == 64 * 48


def test_milidetik_dikonversi_ke_detik(make_run_dir):
    run = read_run(make_run_dir())
    assert run.total == pytest.approx(1.2)
    assert run.stages["baca"].time == pytest.approx(0.120)
    assert run.stages["inisialisasi"].time == pytest.approx(0.001)
    assert run.stages["kmeans"].time == pytest.approx(0.800)
    assert run.stages["pewarnaan"].time == pytest.approx(0.060)
    assert run.stages["tulis"].time == pytest.approx(0.200)
    # Rincian CUDA juga dalam detik, hanya di tahap kmeans.
    assert run.stages["kmeans"].detail == pytest.approx(
        {"h2d": 0.020, "kernel": 0.700, "d2h": 0.010}
    )
    assert run.stages["baca"].detail is None


def test_perangkat_turunan_untuk_mode_cpu(make_run_dir):
    run = read_run(make_run_dir(meta={"mode": "cpu", "kmeans_device": "serial"}))
    assert {s: st.device for s, st in run.stages.items()} == dict.fromkeys(STAGES, "serial")
    assert run.stages["kmeans"].detail is None
    assert validate(run) == []


def test_perangkat_turunan_untuk_mode_opt_omp(make_run_dir):
    run = read_run(make_run_dir(meta={"mode": "opt", "kmeans_device": "omp"}))
    assert {s: st.device for s, st in run.stages.items()} == {
        "baca": "omp", "inisialisasi": "serial", "kmeans": "omp",
        "pewarnaan": "omp", "tulis": "omp",
    }
    assert run.stages["kmeans"].detail is None  # rincian hanya untuk CUDA


def test_perangkat_turunan_untuk_mode_opt_cuda(make_run_dir):
    run = read_run(make_run_dir())  # opt + cuda, tanpa 'devices'
    assert {s: st.device for s, st in run.stages.items()} == {
        "baca": "omp", "inisialisasi": "serial", "kmeans": "cuda",
        "pewarnaan": "omp", "tulis": "omp",
    }


def test_devices_eksplisit_dipetakan_ke_nama_tahap(make_run_dir):
    devices = {"read": "omp", "init": "serial", "kmeans": "cuda", "color": "cuda", "write": "omp"}
    run = read_run(make_run_dir(timing={"devices": devices}))
    assert run.stages["pewarnaan"].device == "cuda"
    assert run.stages["baca"].device == "omp"


def test_devices_sebagian_dilengkapi_dari_turunan(make_run_dir):
    run = read_run(make_run_dir(timing={"devices": {"color": "cuda"}}))
    assert run.stages["pewarnaan"].device == "cuda"
    assert run.stages["baca"].device == "omp"  # dari turunan mode opt


def test_counts_nol_diterima(make_run_dir):
    """Kontrak U2: K lebih besar dari jumlah warna unik, cluster berjumlah 0 sah."""
    labels = np.zeros((48, 64), dtype=np.int64)
    run = read_run(make_run_dir(labels=labels))
    assert list(run.counts) == [64 * 48, 0, 0, 0]
    assert validate(run) == []


def test_jumlah_tahap_boleh_melebihi_total(make_run_dir):
    """Pada L3 tahap tumpang tindih; jangan pernah menambah cek total >= jumlah tahap."""
    run = read_run(make_run_dir(timing={"total_ms": 500.0}))
    assert sum(s.time for s in run.stages.values()) > run.total
    assert validate(run) == []


def test_file_opsional_tidak_wajib(make_run_dir):
    folder = make_run_dir()
    assert not (folder / "map.png").exists() and not (folder / "labels.u8").exists()
    assert validate(read_run(folder)) == []


def test_meta_utf8_dibaca_benar(make_run_dir):
    run = read_run(make_run_dir())
    assert "≥" in run.meta["host"]


def test_json_dengan_bom_diterima(make_run_dir):
    folder = make_run_dir()
    f = folder / "counts.json"
    f.write_bytes(b"\xef\xbb\xbf" + f.read_bytes())
    assert validate(read_run(folder)) == []


def test_png_mode_p_dan_rgba_dikonversi(make_run_dir):
    folder = make_run_dir()
    labels = read_run(folder).labels_preview
    Image.fromarray(labels).convert("P").save(folder / "labels_preview.png")
    rgba = Image.new("RGBA", (64, 48), (10, 20, 30, 255))
    rgba.save(folder / "map_preview.png")
    run = read_run(folder)
    assert run.labels_preview.dtype == np.uint8
    assert np.array_equal(run.labels_preview, labels)
    assert run.map_preview.shape == (48, 64, 3)
    assert tuple(run.map_preview[0, 0]) == (10, 20, 30)


# --------------------------------------------------------------------------- #
# validate(): folder terbaca tetapi isinya tidak masuk akal
# --------------------------------------------------------------------------- #

def test_ditolak_jumlah_counts_salah(make_run_dir):
    problems = validate(read_run(make_run_dir(counts=[1, 2, 3, 4])))
    assert has(problems, "jumlah counts", "10", str(64 * 48))


def test_ditolak_counts_negatif(make_run_dir):
    problems = validate(read_run(make_run_dir(counts=[-1, 64 * 48 + 1, 0, 0])))
    assert has(problems, "negatif")


def test_ditolak_k_tidak_cocok_dengan_centroid_dan_counts(make_run_dir):
    problems = validate(read_run(make_run_dir(meta={"k": 5})))
    assert has(problems, "centroid", "tidak sama dengan k (5)")
    assert has(problems, "counts", "tidak sama dengan k (5)")


def test_ditolak_centroid_dan_counts_beda_panjang(make_run_dir):
    problems = validate(read_run(make_run_dir(counts=[1000, 2072, 0])))
    assert has(problems, "counts (3)", "k (4)")


def test_ditolak_perangkat_tahap_tidak_dikenal(make_run_dir):
    run = read_run(make_run_dir(timing={"devices": {"kmeans": "tpu"}}))
    assert has(validate(run), "kmeans", "tpu", "tidak dikenal")


def test_ditolak_kmeans_device_di_meta_tidak_dikenal(make_run_dir):
    run = read_run(make_run_dir(meta={"kmeans_device": "tpu"}))
    assert has(validate(run), "kmeans_device", "tpu")


def test_ditolak_mode_tidak_dikenal(make_run_dir):
    run = read_run(make_run_dir(meta={"mode": "turbo"}))
    assert has(validate(run), "mode", "turbo")


def test_ditolak_centroid_di_luar_0_255(make_run_dir):
    bad = [[300, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, -5]]
    assert has(validate(read_run(make_run_dir(centroids=bad))), "0-255")


def test_ditolak_id_cluster_preview_melebihi_k(make_run_dir):
    labels = np.zeros((48, 64), dtype=np.int64)
    labels[0, 0] = 9
    problems = validate(read_run(make_run_dir(labels=labels)))
    assert has(problems, "labels_preview", "9", "k = 4")


def test_ditolak_ukuran_map_preview_beda(make_run_dir):
    folder = make_run_dir()
    Image.new("RGB", (32, 24)).save(folder / "map_preview.png")
    assert has(validate(read_run(folder)), "map_preview", "labels_preview")


def test_ditolak_contract_version_tidak_dikenal(make_run_dir):
    run = read_run(make_run_dir(meta={"contract_version": 2}))
    assert has(validate(run), "contract_version", "2")


@pytest.mark.parametrize("total_ms", [0.0, -5.0])
def test_ditolak_total_tidak_positif(make_run_dir, total_ms):
    run = read_run(make_run_dir(timing={"total_ms": total_ms}))
    assert has(validate(run), "total harus > 0")


def test_ditolak_waktu_tahap_negatif(make_run_dir):
    run = read_run(make_run_dir(timing={"write_ms": -1.0}))
    assert has(validate(run), "tulis", "tidak sah")


def test_ditolak_nama_tahap_tidak_dikenal(make_run_dir):
    run = read_run(make_run_dir())
    stages = dict(run.stages)
    stages["lain"] = StageTiming(1.0, "serial")
    bad = dataclasses.replace(run, stages=stages)
    assert has(validate(bad), "nama tahap tidak dikenal", "lain")


def test_validate_memeriksa_file_wajib_di_disk(make_run_dir):
    folder = make_run_dir()
    run = read_run(folder)
    (folder / "map_preview.png").unlink()
    assert has(validate(run), "file wajib tidak ada", "map_preview.png")


def test_validate_mengumpulkan_semua_masalah(make_run_dir):
    run = read_run(make_run_dir(counts=[1, 2, 3, 4], meta={"contract_version": 9}))
    assert len(validate(run)) >= 2


def test_nodata_opsional_dan_valid(make_run_dir):
    old = read_run(make_run_dir())
    assert old.nodata_pixels == 0 and old.nodata_cluster == -1 and validate(old) == []
    with_nd = read_run(make_run_dir(meta={"nodata_pixels": 50, "nodata_cluster": 1}))
    assert with_nd.nodata_pixels == 50 and with_nd.nodata_cluster == 1 and validate(with_nd) == []
    no_nd = read_run(make_run_dir(meta={"nodata_pixels": 0, "nodata_cluster": -1}))
    assert no_nd.nodata_pixels == 0 and no_nd.nodata_cluster == -1 and validate(no_nd) == []


@pytest.mark.parametrize("meta_patch, fragment", [
    ({"nodata_pixels": -1, "nodata_cluster": -1}, "negatif"),
    ({"nodata_pixels": 0, "nodata_cluster": 0}, "harus -1"),
    ({"nodata_pixels": 10, "nodata_cluster": -1}, "di luar rentang"),
    ({"nodata_pixels": 10, "nodata_cluster": 4}, "di luar rentang"),
    ({"nodata_pixels": 769, "nodata_cluster": 0}, "melebihi"),
])
def test_nodata_tidak_konsisten_ditolak_validate(make_run_dir, meta_patch, fragment):
    run = read_run(make_run_dir(meta=meta_patch))
    assert has(validate(run), fragment)


@pytest.mark.parametrize("meta_patch", [
    {"nodata_pixels": "5"}, {"nodata_pixels": True}, {"nodata_cluster": 1.5}, {"nodata_cluster": False},
])
def test_nodata_tipe_salah_ditolak_read_run(make_run_dir, meta_patch):
    with pytest.raises(ContractError):
        read_run(make_run_dir(meta=meta_patch))


# --------------------------------------------------------------------------- #
# read_run(): struktur rusak -> ContractError
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", RUN_FILES)
def test_file_hilang_ditolak_dengan_nama_file(make_run_dir, name):
    folder = make_run_dir(drop=(name,))
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "file wajib tidak ada", name)
    assert name in str(exc.value)


def test_beberapa_file_hilang_dilaporkan_sekaligus(make_run_dir):
    folder = make_run_dir(drop=("counts.json", "timing.json"))
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "counts.json")
    assert has(exc.value.problems, "timing.json")


def test_folder_tidak_ada(tmp_path):
    with pytest.raises(ContractError, match="tidak ditemukan"):
        read_run(tmp_path / "tidak_ada")


def test_json_rusak(make_run_dir):
    folder = make_run_dir()
    (folder / "counts.json").write_text("[1, 2,", encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "counts.json", "JSON")


def test_json_null_ditolak_dengan_pesan_jelas(make_run_dir):
    folder = make_run_dir()
    (folder / "meta.json").write_text("null", encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "meta.json", "null")


@pytest.mark.parametrize("key", ["contract_version", "k", "width", "height", "mode", "kmeans_device"])
def test_kunci_meta_wajib_bukan_keyerror(make_run_dir, key):
    folder = make_run_dir(meta={key: None})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "meta.json", f"'{key}'")


def test_tipe_meta_salah(make_run_dir):
    folder = make_run_dir(meta={"k": "4"})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "'k'", "bilangan bulat")


@pytest.mark.parametrize(
    "key", ["read_ms", "init_ms", "kmeans_ms", "color_ms", "write_ms", "total_ms"]
)
def test_kunci_timing_wajib(make_run_dir, key):
    folder = make_run_dir(timing={key: None})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "timing.json", f"'{key}'")


def test_timing_bukan_angka(make_run_dir):
    folder = make_run_dir(timing={"kmeans_ms": "cepat"})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "kmeans_ms", "angka")


def test_unit_timing_selain_ms_ditolak(make_run_dir):
    folder = make_run_dir(timing={"unit": "s"})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "unit", "'ms'")


def test_kunci_devices_tidak_dikenal(make_run_dir):
    folder = make_run_dir(timing={"devices": {"reed": "omp"}})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "devices", "reed")


def test_centroid_bukan_k_kali_3(make_run_dir):
    folder = make_run_dir(centroids=[[1, 2], [3, 4], [5, 6], [7, 8]])
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "centroids.json", "[r, g, b]")


def test_centroid_pecahan_ditolak(make_run_dir):
    folder = make_run_dir(centroids=[[1.5, 2, 3]] * 4)
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "centroids.json")


def test_labels_preview_berwarna_ditolak(make_run_dir):
    folder = make_run_dir()
    Image.new("RGB", (64, 48)).save(folder / "labels_preview.png")
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "labels_preview.png", "grayscale")


def test_png_rusak(make_run_dir):
    folder = make_run_dir()
    (folder / "map_preview.png").write_bytes(b"bukan png")
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert has(exc.value.problems, "map_preview.png", "PNG")


def test_beberapa_masalah_struktur_dikumpulkan(make_run_dir):
    folder = make_run_dir(drop=("counts.json",), meta={"k": None}, timing={"read_ms": None})
    with pytest.raises(ContractError) as exc:
        read_run(folder)
    assert len(exc.value.problems) >= 3


# --------------------------------------------------------------------------- #
# Batch (plan 4.4)
# --------------------------------------------------------------------------- #

def make_batch(tmp_path, regions=("kulonprogo", "sleman"), batch=None, name="batch"):
    folder = tmp_path / name
    folder.mkdir()
    for r in regions:
        write_run_dir(folder / r)
    data = {"regions": list(regions), "total_ms": 9800.0, "pipelined": True}
    data.update(batch or {})
    (folder / "batch.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


def test_batch_valid(tmp_path):
    batch = read_batch(make_batch(tmp_path))
    assert isinstance(batch, BatchResult)
    assert batch.names == ("kulonprogo", "sleman")
    assert len(batch.runs) == 2 and all(isinstance(r, RunResult) for r in batch.runs)
    assert batch.total == pytest.approx(9.8)
    assert batch.pipelined is True
    assert batch.mock is True
    assert validate_batch(batch) == []


def test_batch_total_cadangan_dianggap_detik(tmp_path):
    folder = make_batch(tmp_path)
    data = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    data.pop("total_ms")
    data["total"] = 9.8  # bentuk di plan 4.4
    (folder / "batch.json").write_text(json.dumps(data), encoding="utf-8")
    assert read_batch(folder).total == pytest.approx(9.8)


def test_batch_tanpa_total_boleh(tmp_path):
    folder = make_batch(tmp_path)
    data = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    data.pop("total_ms")
    (folder / "batch.json").write_text(json.dumps(data), encoding="utf-8")
    assert read_batch(folder).total is None


def test_batch_wilayah_rusak_diberi_awalan_nama(tmp_path):
    folder = make_batch(tmp_path)
    (folder / "sleman" / "counts.json").unlink()
    with pytest.raises(ContractError) as exc:
        read_batch(folder)
    assert has(exc.value.problems, "[sleman]", "counts.json")


@pytest.mark.parametrize("regions", [[], ["../luar"], [".."], [""]])
def test_batch_daftar_wilayah_tidak_sah(tmp_path, regions):
    folder = make_batch(tmp_path, regions=("ada",), batch={"regions": regions})
    with pytest.raises(ContractError):
        read_batch(folder)


def test_batch_json_hilang(tmp_path):
    folder = tmp_path / "kosong"
    folder.mkdir()
    with pytest.raises(ContractError, match="batch.json"):
        read_batch(folder)


def test_validate_batch_melaporkan_wilayah_bermasalah(tmp_path):
    folder = make_batch(tmp_path)
    write_run_dir(folder / "sleman", counts=[1, 2, 3, 4])
    problems = validate_batch(read_batch(folder))
    assert has(problems, "[sleman]", "jumlah counts")
    assert not has(problems, "[kulonprogo]")


def test_read_any_memilih_batch_atau_run(tmp_path, make_run_dir):
    assert isinstance(read_any(make_batch(tmp_path)), BatchResult)
    assert isinstance(read_any(make_run_dir()), RunResult)
