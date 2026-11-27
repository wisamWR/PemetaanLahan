"""Tes verifikasi antarmode (`landcover/verify.py` dan teks vonis `ui/verify_badge.py`, plan Langkah 9).

Run buatan tangan 5x3 (`make_run_dir`, 15 piksel) dipakai untuk semua kasus tepi; `MockLauncher`
dipakai untuk "Cek" plan: mock normal = identik, `break_verification` = berbeda, K berbeda = tidak
sebanding. Tes memeriksa `status` dan kolom `Verification`, bukan kata-kata; kata-kata vonis hanya
diuji di bagian teks (lencana).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from conftest import default_labels
from landcover import verify
from landcover.contract import read_run
from landcover.verify import (
    DIFF_CENTROIDS,
    DIFF_COUNTS,
    DIFF_LABELS,
    SOURCE_FILE,
    SOURCE_LABELS,
    SOURCE_PREVIEW,
    STATUS_DIFFERENT,
    STATUS_IDENTICAL,
    STATUS_NOT_COMPARABLE,
    comparability_problems,
    count_label_differences,
    verification_stamp,
    verify_pair,
)
from launchers.base import RunParams, out_dir_for
from launchers.mock import MockLauncher
from ui import verify_badge as vb

W, H, K = 5, 3, 2
N = W * H
CPU_META = {"mode": "cpu", "kmeans_device": "serial", "threads": 1}
_DROP = object()


# --- pembantu ---------------------------------------------------------------------- #

def make(make_run_dir, name, *, mode="opt", labels=None, meta=None, **kw):
    """Run 5x3 K=2; `mode` menentukan cpu (acuan) atau opt (uji)."""
    merged = {**(CPU_META if mode == "cpu" else {}), **(meta or {})}
    kw.setdefault("width", W)
    kw.setdefault("height", H)
    kw.setdefault("k", K)
    return read_run(make_run_dir(name, labels=labels, meta=merged, **kw))


def pair(make_run_dir, *, cpu_labels=None, opt_labels=None, cpu_kw=None, opt_kw=None):
    cpu = make(make_run_dir, "cpu_run", mode="cpu", labels=cpu_labels, **(cpu_kw or {}))
    opt = make(make_run_dir, "opt_run", mode="opt", labels=opt_labels, **(opt_kw or {}))
    return cpu, opt


def write_labels(run, array) -> None:
    (run.path / "labels.u8").write_bytes(np.asarray(array, dtype=np.uint8).tobytes())


def write_both_labels(cpu, opt, flipped: tuple[int, ...] = ()) -> None:
    flat = default_labels(W, H, K).ravel().astype(np.uint8)
    write_labels(cpu, flat)
    other = flat.copy()
    for i in flipped:
        other[i] ^= 1
    write_labels(opt, other)


def write_vj(run, **override) -> None:
    data = {"ref": "runs/mock/cpu_run", "identical_pct": 100.0, "n_pixels": N, "centroids_equal": True}
    for key, value in override.items():
        if value is _DROP:
            data.pop(key, None)
        else:
            data[key] = value
    (run.path / "verification.json").write_text(json.dumps(data), encoding="utf-8")


def swapped_labels() -> np.ndarray:
    """Dua piksel berlabel beda bertukar tempat: label berbeda, counts tetap sama."""
    arr = default_labels(W, H, K).copy()
    arr[0, 0], arr[0, 3] = arr[0, 3], arr[0, 0]
    return arr


def mock_pair(tmp_path, *, k=8, opt_launcher=None, tile=False, opt_kmeans="omp"):
    data, runs_dir = tmp_path / "data", tmp_path / "runs"
    (data / "uji").mkdir(parents=True)
    if tile:
        (data / "uji" / "manifest.json").write_text(
            json.dumps({"width": 10980, "height": 10980, "pixel_size_m": 10}), encoding="utf-8")
    runs = []
    for mode, kmeans, threads, launcher in (
        ("cpu", "auto", 1, MockLauncher(data, max_side=64, sleep=0)),
        ("opt", opt_kmeans, 4, opt_launcher or MockLauncher(data, max_side=64, sleep=0)),
    ):
        p = RunParams("uji", mode, kmeans, k, 20, 42, threads)
        runs.append(read_run(launcher.run(p, out_dir_for(runs_dir, "mock", p))))
    return runs


# --- sumber pratinjau ------------------------------------------------------------- #

def test_pratinjau_identik_adalah_verifikasi_parsial(make_run_dir):
    cpu, opt = pair(make_run_dir)
    v = verify_pair(cpu, opt)
    assert (v.status, v.source, v.full) == (STATUS_IDENTICAL, SOURCE_PREVIEW, False)
    assert (v.n_compared, v.n_diff, v.identical_pct) == (N, 0, 100.0)
    assert (v.centroids_equal, v.counts_equal) == (True, True)
    assert v.differences == () and v.notes == () and v.reasons == ()
    assert v.mock is True                                    # conftest: meta mock = True


def test_pratinjau_satu_piksel_berbeda(make_run_dir):
    flipped = default_labels(W, H, K).copy()
    flipped[2, 4] ^= 1
    cpu, opt = pair(make_run_dir, opt_labels=flipped)
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.source == SOURCE_PREVIEW
    assert (v.n_compared, v.n_diff) == (N, 1)
    assert v.identical_pct == pytest.approx(100.0 * 14 / 15) and v.identical_pct < 100
    assert set(v.differences) == {DIFF_LABELS, DIFF_COUNTS}  # counts.json opt dihitung dari labelnya
    assert v.centroids_equal is True and v.counts_equal is False


def test_label_berbeda_dengan_counts_sama(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_labels=swapped_labels())
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT
    assert v.differences == (DIFF_LABELS,)
    assert v.n_diff == 2 and v.counts_equal is True and v.centroids_equal is True


def test_hanya_centroid_berbeda(make_run_dir):
    other = [[34, 61, 40], [201, 198, 191]]
    cpu, opt = pair(make_run_dir, opt_kw={"centroids": other})
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.differences == (DIFF_CENTROIDS,)
    assert v.n_diff == 0 and v.identical_pct == 100.0 and v.centroids_equal is False


def test_hanya_counts_berbeda_ditandai_tidak_konsisten(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"counts": [7, 8]})     # counts.json != hitungan label
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.differences == (DIFF_COUNTS,)
    assert v.n_diff == 0
    assert any("tidak konsisten" in n for n in v.notes)


def test_pratinjau_beda_ukuran_tidak_dibandingkan_tetapi_counts_dan_centroid_ya(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_labels=default_labels(10, 6, K), opt_kw={"counts": [8, 7]})
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_IDENTICAL and v.source == SOURCE_PREVIEW
    assert (v.n_compared, v.n_diff, v.identical_pct) == (None, None, None)
    assert any("berbeda ukuran" in n for n in v.notes)
    # counts berbeda tetap tertangkap walau label pratinjau tidak bisa dibandingkan
    cpu2, opt2 = pair(make_run_dir, opt_labels=default_labels(10, 6, K))
    v2 = verify_pair(cpu2, opt2)
    assert v2.status == STATUS_DIFFERENT and v2.differences == (DIFF_COUNTS,)


# --- sebanding ---------------------------------------------------------------------- #

def test_k_berbeda_tidak_sebanding(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"k": 3})
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_NOT_COMPARABLE and v.source is None
    assert any("K berbeda (2 vs 3)" in r for r in v.reasons)
    assert (v.n_compared, v.n_diff, v.identical_pct, v.centroids_equal, v.counts_equal) == (None,) * 5


@pytest.mark.parametrize("opt_kw, needle", [
    ({"meta": {"dataset": "lain"}}, "dataset berbeda"),
    ({"meta": {"iters": 5}}, "iterasi berbeda"),
    ({"meta": {"seed": 7}}, "seed berbeda"),
    ({"width": 6}, "ukuran citra berbeda (5×3 vs 6×3)"),
    ({"height": 4}, "ukuran citra berbeda (5×3 vs 5×4)"),
    ({"meta": {"mock": False}}, "tiruan"),
])
def test_parameter_berbeda_tidak_sebanding(make_run_dir, opt_kw, needle):
    cpu, opt = pair(make_run_dir, opt_kw=opt_kw)
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_NOT_COMPARABLE
    assert any(needle in r for r in v.reasons), v.reasons


def test_semua_alasan_dikumpulkan(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"k": 3, "meta": {"seed": 1}})
    assert len(verify_pair(cpu, opt).reasons) == 2


def test_mode_harus_cpu_lalu_opt(make_run_dir):
    cpu, opt = pair(make_run_dir)
    assert verify_pair(opt, cpu).status == STATUS_NOT_COMPARABLE        # terbalik
    assert verify_pair(opt, opt).status == STATUS_NOT_COMPARABLE        # dua opt
    assert verify_pair(cpu, cpu).status == STATUS_NOT_COMPARABLE        # dua cpu
    assert comparability_problems(cpu, opt) == []


def test_tidak_sebanding_tidak_membaca_berkas_label(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"k": 3})
    write_labels(cpu, np.zeros(N)), write_labels(opt, np.zeros(3))      # rusak, tak boleh disentuh
    assert verify_pair(cpu, opt).status == STATUS_NOT_COMPARABLE


# --- sumber labels.u8 --------------------------------------------------------------- #

def test_labels_identik(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt)
    v = verify_pair(cpu, opt)
    assert (v.status, v.source, v.full) == (STATUS_IDENTICAL, SOURCE_LABELS, True)
    assert (v.n_compared, v.n_diff, v.identical_pct) == (N, 0, 100.0)
    assert v.notes == ()


@pytest.mark.parametrize("position", [0, 3, 4, 7, 8, 11, 12, 14])
def test_satu_label_berbeda_di_posisi_mana_pun_terdeteksi_lintas_blok(make_run_dir, position):
    """block_bytes=4 membagi 15 byte menjadi blok 4+4+4+3: tepi blok dan blok terakhir ikut diuji."""
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt, flipped=(position,))
    v = verify_pair(cpu, opt, block_bytes=4)
    assert v.status == STATUS_DIFFERENT and v.source == SOURCE_LABELS
    assert (v.n_compared, v.n_diff) == (N, 1)
    assert v.identical_pct == pytest.approx(100.0 * 14 / 15)
    assert v.differences == (DIFF_LABELS,)                  # pratinjau dan counts tetap sama


def test_beberapa_label_berbeda_dihitung_semua(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt, flipped=(0, 5, 14))
    v = verify_pair(cpu, opt, block_bytes=4)
    assert v.n_diff == 3 and v.identical_pct == pytest.approx(80.0)


def test_labels_dipakai_walau_pratinjau_identik(make_run_dir):
    """Label resolusi penuh bisa berbeda walau pratinjau (sampling) kebetulan sama."""
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt, flipped=(6,))
    assert verify_pair(cpu, opt).status == STATUS_DIFFERENT


def test_ukuran_labels_salah_dilewati(make_run_dir):
    cpu, opt = pair(make_run_dir)
    flat = default_labels(W, H, K).ravel()
    write_labels(cpu, flat)
    write_labels(opt, flat[:-1])                            # 14 byte, seharusnya 15
    v = verify_pair(cpu, opt)
    assert v.source == SOURCE_PREVIEW and v.status == STATUS_IDENTICAL
    assert any("berukuran 14 byte" in n and "opt_run" in n for n in v.notes)


def test_hanya_satu_labels_dilewati(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_labels(opt, default_labels(W, H, K))
    v = verify_pair(cpu, opt)
    assert v.source == SOURCE_PREVIEW
    assert any("hanya ada di folder 'opt_run'" in n for n in v.notes)


def test_label_identik_tetapi_counts_berbeda_bukan_identik(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"counts": [7, 8]})
    write_both_labels(cpu, opt)
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.differences == (DIFF_COUNTS,)
    assert any("tidak konsisten" in n for n in v.notes)


def test_berkas_label_tidak_terkunci_setelah_verifikasi(make_run_dir):
    """Di Windows pemetaan yang tertinggal mengunci berkas dan membuat os.replace/hapus gagal."""
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt)
    verify_pair(cpu, opt)
    tmp = opt.path / "labels.u8.tmp"
    tmp.write_bytes(b"\x00" * N)
    os.replace(tmp, opt.path / "labels.u8")
    (cpu.path / "labels.u8").unlink()


def test_count_label_differences(tmp_path):
    a, b = tmp_path / "a.u8", tmp_path / "b.u8"
    a.write_bytes(bytes(range(10)))
    b.write_bytes(bytes([0, 1, 9, 3, 4, 5, 6, 7, 0, 0]))     # beda di indeks 2, 8, 9
    for block in (1, 3, 4, 10, 11, 1 << 24):
        assert count_label_differences(a, b, 10, block_bytes=block) == 3
    assert count_label_differences(a, b, 0) == 0
    assert count_label_differences(a, b, 2) == 0            # hanya 2 byte pertama dibandingkan
    with pytest.raises(OSError):
        count_label_differences(a, b, 11)                   # berkas lebih pendek dari yang diminta
    with pytest.raises(ValueError):
        count_label_differences(a, b, 10, block_bytes=0)
    with pytest.raises(OSError):
        count_label_differences(tmp_path / "tidak_ada", b, 10)


# --- sumber verification.json -------------------------------------------------------- #

def test_verification_json_valid(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt)
    v = verify_pair(cpu, opt)
    assert (v.status, v.source, v.full) == (STATUS_IDENTICAL, SOURCE_FILE, True)
    assert (v.n_compared, v.n_diff, v.identical_pct) == (N, 0, 100.0) and v.notes == ()


def test_verification_json_didahulukan_dari_labels(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt, flipped=(1,))               # labels.u8 berbeda, tetapi urutan sumber: berkas dulu
    write_vj(opt)
    v = verify_pair(cpu, opt)
    assert v.source == SOURCE_FILE and v.status == STATUS_IDENTICAL


@pytest.mark.parametrize("ref", [
    "runs/mock/cpu_run", "runs/mock/cpu_run/", "cpu_run", r"D:\projek\runs\mock\cpu_run",
    "/mnt/d/projek/runs/mock/cpu_run", "  runs/mock/cpu_run  ",
])
def test_ref_dicocokkan_lewat_nama_folder_bukan_path_penuh(make_run_dir, ref):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, ref=ref)
    assert verify_pair(cpu, opt).source == SOURCE_FILE


def test_persen_di_bawah_100_dari_berkas_berbeda(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, identical_pct=99.98)
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.source == SOURCE_FILE
    assert v.differences == (DIFF_LABELS,) and v.identical_pct == 99.98 and v.n_diff is None


def test_persen_hampir_100_tidak_dianggap_identik(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, identical_pct=99.99999999)
    assert verify_pair(cpu, opt).status == STATUS_DIFFERENT


def test_berkas_tidak_ada_bukan_masalah(make_run_dir):
    cpu, opt = pair(make_run_dir)
    assert verify_pair(cpu, opt).notes == ()


def test_berkas_dengan_bom_dibaca(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt)
    path = opt.path / "verification.json"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert verify_pair(cpu, opt).source == SOURCE_FILE


@pytest.mark.parametrize("case, content", [
    ("json rusak", "{tidak json"),
    ("kosong", ""),
    ("bukan objek", "[1, 2]"),
    ("null", "null"),
    ("bukan utf-8", b"\xff\xfe\x00"),
])
def test_berkas_rusak_dilewati_dengan_catatan(make_run_dir, case, content):
    cpu, opt = pair(make_run_dir)
    path = opt.path / "verification.json"
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    v = verify_pair(cpu, opt)                               # tidak melempar
    assert v.source == SOURCE_PREVIEW and v.status == STATUS_IDENTICAL
    assert len(v.notes) == 1 and "verification.json" in v.notes[0] and "tidak dipakai" in v.notes[0]


@pytest.mark.parametrize("field, value", [
    ("centroids_equal", "true"), ("centroids_equal", 1), ("centroids_equal", None), ("centroids_equal", _DROP),
    ("identical_pct", float("nan")), ("identical_pct", float("inf")), ("identical_pct", -0.1),
    ("identical_pct", 100.5), ("identical_pct", True), ("identical_pct", "100"), ("identical_pct", _DROP),
    ("n_pixels", N + 1), ("n_pixels", float(N)), ("n_pixels", True), ("n_pixels", "15"), ("n_pixels", _DROP),
    ("ref", "runs/mock/lain"), ("ref", ""), ("ref", "   "), ("ref", 7), ("ref", _DROP),
])
def test_berkas_salah_bentuk_atau_bukan_pasangan_ini_dilewati(make_run_dir, field, value):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, **{field: value})
    v = verify_pair(cpu, opt)
    assert v.source == SOURCE_PREVIEW and v.status == STATUS_IDENTICAL
    assert len(v.notes) == 1 and "tidak dipakai" in v.notes[0]


def test_ref_yang_salah_menyebut_kedua_nama(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, ref="runs/mock/lain")
    (note,) = verify_pair(cpu, opt).notes
    assert "lain" in note and "cpu_run" in note


def test_berkas_tidak_dipakai_lalu_labels_dipakai(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, ref="runs/mock/lain")
    write_both_labels(cpu, opt, flipped=(2,))
    v = verify_pair(cpu, opt)
    assert v.source == SOURCE_LABELS and v.status == STATUS_DIFFERENT and v.n_diff == 1
    assert len(v.notes) == 1 and "verification.json" in v.notes[0]


def test_berkas_menyatakan_identik_tetapi_counts_berbeda(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"counts": [7, 8]})
    write_vj(opt)
    v = verify_pair(cpu, opt)
    assert v.source == SOURCE_FILE and v.status == STATUS_DIFFERENT and v.differences == (DIFF_COUNTS,)


def test_berkas_dan_centroids_json_tidak_sepakat(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"centroids": [[34, 61, 40], [201, 198, 191]]})
    write_vj(opt, centroids_equal=True)                     # berkas bilang sama, centroids.json bilang beda
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.differences == (DIFF_CENTROIDS,) and v.centroids_equal is False
    assert any("tidak sepakat" in n for n in v.notes)

    cpu2, opt2 = pair(make_run_dir)
    write_vj(opt2, centroids_equal=False)                   # sebaliknya: bukti negatif tetap menang
    v2 = verify_pair(cpu2, opt2)
    assert v2.status == STATUS_DIFFERENT and v2.differences == (DIFF_CENTROIDS,)
    assert any("tidak sepakat" in n for n in v2.notes)


# --- persen dan kunci cache ------------------------------------------------------------ #

def test_satu_piksel_dari_tile_penuh_bukan_100_persen():
    pct = verify._pct(1, 120_560_400)
    assert pct < 100.0
    assert verify._pct(0, 120_560_400) == 100.0
    assert verify._pct(1, 10**18) < 100.0                    # selisih hilang di float: dipaksa di bawah 100
    assert verify._pct(1, 15) == pytest.approx(100.0 * 14 / 15)


def test_stamp_berubah_bila_berkas_yang_dibaca_berubah(make_run_dir):
    cpu, _ = pair(make_run_dir)
    base = verification_stamp(cpu)
    assert verification_stamp(cpu) == base
    assert ("path", str(cpu.path)) in base
    write_labels(cpu, np.zeros(N))
    with_labels = verification_stamp(cpu)
    assert with_labels != base
    write_labels(cpu, np.ones(N))
    os.utime(cpu.path / "labels.u8", ns=(1, 1))              # isi dan waktu ubah berbeda, ukuran sama
    assert verification_stamp(cpu) not in (base, with_labels)
    (cpu.path / "labels.u8").unlink()
    assert verification_stamp(cpu) == base
    write_vj(cpu)
    assert verification_stamp(cpu) != base
    counts = cpu.path / "counts.json"
    before = verification_stamp(cpu)
    os.utime(counts, ns=(2, 2))
    assert verification_stamp(cpu) != before


def test_stamp_run_berbeda_berbeda(make_run_dir):
    cpu, opt = pair(make_run_dir)
    assert verification_stamp(cpu) != verification_stamp(opt)


# --- "Cek" plan dengan MockLauncher --------------------------------------------------------- #

def test_mock_normal_identik(tmp_path):
    cpu, opt = mock_pair(tmp_path)
    v = verify_pair(cpu, opt)
    assert (v.status, v.source) == (STATUS_IDENTICAL, SOURCE_PREVIEW)
    assert v.n_diff == 0 and v.centroids_equal and v.counts_equal and v.mock


def test_mock_break_verification_berbeda(tmp_path):
    data = tmp_path / "data"
    broken = MockLauncher(data, max_side=64, sleep=0, break_verification=True)
    cpu, opt = mock_pair(tmp_path, opt_launcher=broken)
    v = verify_pair(cpu, opt)
    assert v.status == STATUS_DIFFERENT and v.source == SOURCE_PREVIEW
    assert DIFF_LABELS in v.differences and v.centroids_equal is True
    assert 98.0 < v.identical_pct < 100.0 and v.n_diff >= 1          # ~1 % label diubah (plan)


def test_mock_k_berbeda_tidak_sebanding(tmp_path):
    cpu, _ = mock_pair(tmp_path, k=8)
    p = RunParams("uji", "opt", "omp", 5, 20, 42, 4)
    opt5 = read_run(MockLauncher(tmp_path / "data", max_side=64, sleep=0).run(p, out_dir_for(tmp_path / "runs", "mock", p)))
    assert verify_pair(cpu, opt5).status == STATUS_NOT_COMPARABLE


# --- teks vonis (lencana) ---------------------------------------------------------------------- #

@pytest.mark.parametrize("pct, text", [
    (100.0, "100"), (100, "100"), (99.99999917, "99.99"), (99.999, "99.99"), (99.995, "99.99"),
    (99.98, "99.98"), (99.29, "99.29"), (50, "50.00"), (0, "0.00"), (1e-05, "0.00"), (93.33333, "93.33"),
])
def test_persen_dipotong_bukan_dibulatkan(pct, text):
    assert vb.format_identical_pct(pct) == text


def test_persen_di_bawah_100_tidak_pernah_tampil_100():
    for pct in (99.999999, math.nextafter(100.0, 0.0), 100.0 * (120_560_400 - 1) / 120_560_400):
        assert vb.format_identical_pct(pct).startswith("99.")


def test_format_piksel():
    assert vb.format_pixels(120_560_400) == "120,560,400" and vb.format_pixels(15) == "15"


def test_vonis_identik_resolusi_penuh_dari_labels_dan_berkas(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_both_labels(cpu, opt)
    v = verify_pair(cpu, opt)
    assert vb.verdict_text(v) == "Label identik 100 % (15 piksel), centroid sama"
    assert vb.verdict_style(v) == ("✅", "green")
    assert "labels.u8" in vb.source_text(v)
    (opt.path / "labels.u8").unlink(), (cpu.path / "labels.u8").unlink()
    write_vj(opt)
    v2 = verify_pair(cpu, opt)
    assert vb.verdict_text(v2) == "Label identik 100 % (15 piksel), centroid sama"
    assert "verification.json" in vb.source_text(v2)


def test_vonis_pratinjau_tidak_pernah_mengaku_resolusi_penuh(tmp_path):
    cpu, opt = mock_pair(tmp_path, tile=True)               # tile 10980x10980 = 120.560.400 piksel, pratinjau 64x64
    assert cpu.n_pixels == 120_560_400
    v = verify_pair(cpu, opt)
    text = vb.verdict_text(v)
    assert text.startswith("Verifikasi parsial")
    assert "4,096 piksel pratinjau" in text
    assert "120,560,400" not in text and "Label identik" not in text and "100 %" not in text
    assert "belum dibandingkan" in text
    assert vb.verdict_style(v) == ("ℹ️", "blue")            # bukan hijau: bukan bukti resolusi penuh
    assert "120,560,400" not in " ".join(r[1] for r in vb.detail_rows(v))
    assert "labels.u8" in vb.source_text(v)


def test_vonis_pratinjau_beda_ukuran(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_labels=default_labels(10, 6, K), opt_kw={"counts": [8, 7]})
    text = vb.verdict_text(verify_pair(cpu, opt))
    assert text.startswith("Verifikasi parsial") and "tidak dibandingkan" in text and "piksel pratinjau" not in text


def test_vonis_berbeda(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_labels=swapped_labels())
    v = verify_pair(cpu, opt)
    assert vb.verdict_text(v) == "Berbeda: 86.66 % label pratinjau identik (2 dari 15 piksel berbeda)"
    assert vb.verdict_style(v) == ("❌", "red")
    write_both_labels(cpu, opt, flipped=(0,))
    v2 = verify_pair(cpu, opt)                              # label resolusi penuh: kata "pratinjau" hilang
    assert vb.verdict_text(v2) == "Berbeda: 93.33 % label identik (1 dari 15 piksel berbeda)"


def test_vonis_berbeda_menyebut_semua_perbedaan(make_run_dir):
    flipped = default_labels(W, H, K).copy()
    flipped[2, 4] ^= 1
    cpu, opt = pair(make_run_dir, opt_labels=flipped, opt_kw={"centroids": [[0, 0, 0], [1, 1, 1]]})
    text = vb.verdict_text(verify_pair(cpu, opt))
    assert "label pratinjau identik" in text and "centroid berbeda" in text and "counts berbeda" in text
    assert text.index("label") < text.index("centroid") < text.index("counts")


def test_vonis_berbeda_hanya_centroid_menyebut_label_identik(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"centroids": [[34, 61, 40], [201, 198, 191]]})
    assert vb.verdict_text(verify_pair(cpu, opt)) == "Berbeda: label pratinjau identik; centroid berbeda"


def test_vonis_dari_berkas_tanpa_jumlah_piksel_berbeda(make_run_dir):
    cpu, opt = pair(make_run_dir)
    write_vj(opt, identical_pct=99.98)
    assert vb.verdict_text(verify_pair(cpu, opt)) == "Berbeda: 99.98 % label identik"


def test_vonis_tidak_sebanding(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_kw={"k": 3})
    v = verify_pair(cpu, opt)
    assert vb.verdict_text(v) == "Tidak sebanding: K berbeda (2 vs 3)"
    assert vb.verdict_style(v) == ("⚠️", "orange")
    assert vb.source_text(v) is None and vb.detail_rows(v) == []


def test_rincian(make_run_dir):
    cpu, opt = pair(make_run_dir, opt_labels=swapped_labels(), opt_kw={"centroids": [[0, 0, 0], [1, 1, 1]]})
    rows = dict(vb.detail_rows(verify_pair(cpu, opt)))
    assert rows["Label pratinjau"].startswith("86.66 % label pratinjau identik")
    assert rows["Centroid"] == "berbeda" and rows["Counts"] == "sama" and rows["Sumber"] == "preview"
    cpu2, opt2 = pair(make_run_dir)
    write_both_labels(cpu2, opt2)
    assert dict(vb.detail_rows(verify_pair(cpu2, opt2)))["Label"] == "identik (15 piksel)"


def test_escape_markdown():
    assert vb.escape_markdown("a_b*c[d]$e`f~g\\h") == "a\\_b\\*c\\[d\\]\\$e\\`f\\~g\\\\h"
    assert vb.escape_markdown("K berbeda (8 vs 5)") == "K berbeda (8 vs 5)"
