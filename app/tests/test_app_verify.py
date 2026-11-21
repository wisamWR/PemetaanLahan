"""Alur layar status verifikasi antarmode dengan AppTest (plan Langkah 9; logikanya di test_verify.py).

Dataset "kecil" = PNG 40x30 (1.200 piksel), pratinjau = ukuran penuh, jadi `labels.u8` buatan tangan
bisa diturunkan dari `labels_preview.png`. Folder data/ dan runs/ sementara (fixture `sandbox`).
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from launchers.base import RunParams, out_dir_for
from launchers.mock import MockLauncher
from test_app_flow import (  # noqa: F401  (sandbox = fixture pytest)
    CPU_ID, OPT_ID, captions, no_errors, press, run_mode, sandbox, start,
)
from ui import verify_badge as vb

N = 40 * 30
TITLE = "Status verifikasi antarmode"


def markdown(at) -> list[str]:
    return [m.value for m in at.markdown]


def verdicts(at) -> list[str]:
    """Teks vonis (markdown berwarna berisi ikon vonis)."""
    return [m for m in markdown(at) if any(i in m for i in (vb.ICON_OK, vb.ICON_PARTIAL, vb.ICON_BAD, vb.ICON_WARN))]


def titles(at) -> list[str]:
    return [s.value for s in at.subheader if s.value.startswith(TITLE)]


def both_runs(at) -> None:
    run_mode(at, "cpu")
    run_mode(at, "opt")
    no_errors(at)


def write_labels_from_preview(run_dir) -> np.ndarray:
    labels = np.array(Image.open(run_dir / "labels_preview.png"), dtype=np.uint8).ravel()
    assert labels.size == N
    (run_dir / "labels.u8").write_bytes(labels.tobytes())
    return labels


def add_broken_opt(at, data, runs) -> str:
    """Daftarkan run opt (OpenMP) yang labelnya sengaja diubah ~1 % dan jadikan aktif."""
    p = RunParams("kecil", "opt", "omp", 8, 20, 42, 8)
    out = out_dir_for(runs, "mock", p)
    MockLauncher(data, sleep=0, break_verification=True).run(p, out)
    state = dict(at.session_state["runs"])
    state[out.name] = str(out)
    at.session_state["runs"] = state
    at.session_state["active_run"] = out.name
    at.run()
    assert not at.exception, at.exception
    return out.name


# --- keadaan awal dan satu sisi --------------------------------------------------------- #

def test_kotak_kosong_sebelum_ada_hasil(sandbox):
    at = start()
    assert f"{TITLE}: belum ada hasil (jalankan dulu)" in captions(at)
    assert titles(at) == [] and verdicts(at) == []


def test_satu_run_saja_meminta_pasangan(sandbox):
    at = start()
    run_mode(at, "cpu")
    no_errors(at)
    assert len(titles(at)) == 1 and "DATA TIRUAN" in titles(at)[0]
    assert "Butuh kedua run untuk membandingkan. Jalankan Optimalisasi untuk kecil · K=8 · 20 iterasi · seed 42 " \
           "(dataset, K, iterasi, dan seed harus sama)." in captions(at)
    assert verdicts(at) == [] and not any("Pembanding:" in c for c in captions(at))
    assert not any(e.label == "Rincian verifikasi" for e in at.expander)


def test_hanya_optimalisasi_meminta_cpu(sandbox):
    at = start()
    run_mode(at, "opt")
    no_errors(at)
    assert any(c.startswith("Butuh kedua run untuk membandingkan. Jalankan CPU saja (L0) untuk kecil") for c in captions(at))
    assert verdicts(at) == []


def test_k_berbeda_tidak_dipasangkan(sandbox):
    at = start()
    run_mode(at, "cpu")                                                  # CPU K=8
    at.sidebar.slider(key="f_k").set_value(5)
    run_mode(at, "opt")                                                  # Optimalisasi K=5: tidak sebanding dengan CPU K=8
    no_errors(at)
    assert any(c.startswith("Butuh kedua run untuk membandingkan. Jalankan CPU saja (L0) untuk kecil · K=5") for c in captions(at))
    assert verdicts(at) == []                                            # tidak ada vonis palsu untuk pasangan beda K


# --- pasangan tiruan --------------------------------------------------------------------- #

def test_pasangan_tiruan_normal_adalah_verifikasi_parsial(sandbox):
    at = start()
    both_runs(at)
    (verdict,) = verdicts(at)
    assert verdict.startswith(":blue[**ℹ️ Verifikasi parsial")
    assert f"{N:,} piksel pratinjau" in verdict and "Label identik" not in verdict
    assert "Pembanding: CPU saja (L0) ↔ Optimalisasi · " in " ".join(captions(at))
    assert vb.MOCK_VERIFY_NOTICE in captions(at)
    assert any("Sumber: pratinjau" in c for c in captions(at))
    assert not any(c.startswith("Catatan:") for c in captions(at))
    assert "DATA TIRUAN" in titles(at)[0]
    (expander,) = [e for e in at.expander if e.label == "Rincian verifikasi"]
    body = " ".join(m.value for m in expander.markdown)
    assert "Centroid" in body and "sama" in body and "Counts" in body


def test_pasangan_dengan_label_diubah_tampil_berbeda_bukan_galat(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    add_broken_opt(at, data, runs)
    no_errors(at)                                                         # vonis "Berbeda" bukan galat aplikasi
    (verdict,) = verdicts(at)
    assert verdict.startswith(":red[**❌ Berbeda:") and "99.00 % label pratinjau identik (12 dari 1,200 piksel berbeda)" in verdict
    assert "counts berbeda" in verdict and "centroid" not in verdict


def test_label_resolusi_penuh_dan_cache_mengikuti_berkas(sandbox):
    data, runs = sandbox
    at = start()
    both_runs(at)
    assert verdicts(at)[0].startswith(":blue[**ℹ️ Verifikasi parsial")

    cpu_dir, opt_dir = runs / "mock" / CPU_ID, runs / "mock" / OPT_ID
    write_labels_from_preview(cpu_dir)
    write_labels_from_preview(opt_dir)
    at.run()
    assert not at.exception
    (verdict,) = verdicts(at)
    assert verdict == f":green[**✅ Label identik 100 % ({N:,} piksel), centroid sama**]"
    assert any("Sumber: labels.u8" in c for c in captions(at))

    flat = np.frombuffer((opt_dir / "labels.u8").read_bytes(), dtype=np.uint8).copy()
    flat[7] = (flat[7] + 1) % 8
    (opt_dir / "labels.u8").write_bytes(flat.tobytes())
    os.utime(opt_dir / "labels.u8", ns=(5, 5))                            # waktu ubah pasti beda dari tulisan sebelumnya
    at.run()
    assert not at.exception
    (verdict,) = verdicts(at)
    assert verdict.startswith(":red[**❌ Berbeda: 99.91 % label identik (1 dari 1,200 piksel berbeda)")
    no_errors(at)

    (opt_dir / "labels.u8").unlink()                                      # satu sisi saja: kembali ke pratinjau + catatan
    at.run()
    (verdict,) = verdicts(at)
    assert verdict.startswith(":blue[**ℹ️ Verifikasi parsial")
    assert any(c.startswith("Catatan:") and "hanya ada di folder" in c for c in captions(at))


def test_verification_json_dipakai_dan_berkas_salah_dicatat(sandbox):
    data, runs = sandbox
    at = start()
    both_runs(at)
    opt_dir = runs / "mock" / OPT_ID
    vj = opt_dir / "verification.json"

    vj.write_text(json.dumps({"ref": f"runs/mock/{CPU_ID}", "identical_pct": 100.0, "n_pixels": N,
                              "centroids_equal": True}), encoding="utf-8")
    at.run()
    assert not at.exception
    assert verdicts(at) == [f":green[**✅ Label identik 100 % ({N:,} piksel), centroid sama**]"]
    assert any("Sumber: verification.json" in c for c in captions(at))

    vj.write_text(json.dumps({"ref": "runs/mock/lain", "identical_pct": 100.0, "n_pixels": N,
                              "centroids_equal": True}), encoding="utf-8")
    at.run()
    assert not at.exception
    assert verdicts(at)[0].startswith(":blue[**ℹ️ Verifikasi parsial")      # berkas bukan untuk pasangan ini: tidak dipercaya
    assert any(c.startswith("Catatan: verification.json merujuk ke folder 'lain'") for c in captions(at))

    vj.write_text("{rusak", encoding="utf-8")
    at.run()
    assert not at.exception                                               # berkas rusak tidak membuat halaman mogok
    assert any(c.startswith("Catatan: verification.json tidak terbaca") for c in captions(at))
    no_errors(at)


def test_catatan_dari_berkas_di_escape_agar_tidak_menjadi_markdown(sandbox):
    data, runs = sandbox
    at = start()
    both_runs(at)
    (runs / "mock" / OPT_ID / "verification.json").write_text(
        json.dumps({"ref": "runs/mock/a$b[c]", "identical_pct": 100.0, "n_pixels": N, "centroids_equal": True}),
        encoding="utf-8")
    at.run()
    assert not at.exception
    (note,) = [c for c in captions(at) if c.startswith("Catatan:")]
    assert "'a\\$b\\[c\\]'" in note and "a$b" not in note       # `$` dan `[` akan memicu LaTeX/tautan bila tidak di-escape


def test_nama_dataset_di_petunjuk_pasangan_di_escape(sandbox):
    import shutil

    data, _ = sandbox
    shutil.copy(data / "kecil.png", data / "uji_a.png")
    at = start()
    at.sidebar.selectbox(key="f_dataset").set_value("uji_a")
    run_mode(at, "cpu")
    no_errors(at)
    (hint,) = [c for c in captions(at) if c.startswith("Butuh kedua run untuk membandingkan")]
    assert "untuk uji\\_a · K=8" in hint                                   # `_` di nama dataset tidak boleh menjadi miring


def test_berkas_basi_dari_run_lama_tidak_dipakai_setelah_run_ulang(sandbox):
    """Optimalisasi selalu dijalankan ulang ke folder yang sama; labels.u8 lama bukan milik run baru."""
    data, runs = sandbox
    at = start()
    both_runs(at)
    cpu_dir, opt_dir = runs / "mock" / CPU_ID, runs / "mock" / OPT_ID
    write_labels_from_preview(cpu_dir)
    write_labels_from_preview(opt_dir)
    at.run()
    assert verdicts(at)[0].startswith(":green[**✅ Label identik 100 %")
    press(at, "btn_compare")                                              # CPU dipakai ulang, Optimalisasi dijalankan lagi
    no_errors(at)
    assert not (opt_dir / "labels.u8").exists() and (cpu_dir / "labels.u8").is_file()
    assert verdicts(at)[0].startswith(":blue[**ℹ️ Verifikasi parsial")      # bukan hijau dari labels.u8 basi
    assert any(c.startswith("Catatan:") and "hanya ada di folder" in c for c in captions(at))


def test_verifikasi_tidak_dihitung_ulang_pada_suntingan_lain(sandbox, monkeypatch):
    """Halaman dijalankan ulang tiap widget berubah; verifikasi (bisa membaca 2x120 MB) harus di-cache."""
    calls: list[int] = []
    original = vb.verify_pair

    def counting(ref, run, **kw):
        calls.append(1)
        return original(ref, run, **kw)

    monkeypatch.setattr(vb, "verify_pair", counting)
    at = start()
    both_runs(at)
    assert len(calls) == 1
    at.run()
    at.sidebar.slider(key="f_k").set_value(5)                             # ganti K tanpa kirim: hasil tetap
    at.run()
    at.sidebar.radio(key="f_mode").set_value("opt")
    at.run()
    assert not at.exception and len(calls) == 1                           # tetap satu kali

    runs_dir = sandbox[1]                                                 # berkas berubah: dihitung ulang
    write_labels_from_preview(runs_dir / "mock" / CPU_ID)
    at.run()
    assert len(calls) == 2


def test_galat_verifikasi_tampil_sebagai_peringatan(sandbox, monkeypatch):
    def boom(ref, run, **kw):
        raise OSError("disk penuh $5 [x]")

    monkeypatch.setattr(vb, "verify_pair", boom)
    at = start()
    both_runs(at)
    assert any("Verifikasi tidak bisa dijalankan: disk penuh \\$5 \\[x\\]" in w.value for w in at.warning)
    assert verdicts(at) == []
    assert len(at.get("image")) == 2                                      # bagian halaman lain tetap tampil
