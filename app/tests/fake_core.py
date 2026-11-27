"""Pengganti program `landcover` untuk tes CliLauncher (BUKAN bagian aplikasi).

Dijalankan sebagai `python fake_core.py <argumen landcover>`. Pada keadaan normal ia meniru program
sungguhan pada citra kecil: membaca satu PNG (atau folder dengan manifest.json), menjalankan K-Means
referensi (`reference_kmeans`), lalu menulis folder run kontrak 4.3 dengan format dan nama berkas
yang sama seperti `src/cli/landcover.cpp`, sehingga lolos `read_run` dan `validate`.

Perilaku diatur lewat variabel lingkungan (diwarisi dari pytest):
  FAKE_CORE         ok (bawaan) | exit1 | exit2 | exit3 | exit7 | silent0 | garbage | badjson |
                    wrongname | mock | hang | bigerr | signal | staleguard | utf8
  FAKE_CORE_DROP    nama berkas yang TIDAK ditulis (mode ok)
  FAKE_ENGINES      daftar dipisah koma dari "omp,cuda" yang dianggap terhubung (bawaan: kosong)
  FAKE_CORE_LOG     berkas JSON-lines: argumen dan isi folder keluaran SAAT proses dimulai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # app/

import numpy as np                                               # noqa: E402
from PIL import Image                                            # noqa: E402

from landcover.reference_kmeans import init_centroids, kmeans_int  # noqa: E402

RUN_FILES = ("meta.json", "timing.json", "centroids.json", "counts.json", "labels_preview.png",
             "map_preview.png", "map.png", "labels.u8", "verification.json")


def engines() -> dict:
    have = {e for e in os.getenv("FAKE_ENGINES", "").split(",") if e}
    return {"cpu": "ref_kmeans", "omp": "omp" in have, "cuda": "cuda" in have}


def load_image(path: Path):
    """(rgb (h, w, 3), pixel_size, chunks) dari satu PNG atau folder manifest.json."""
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        w, h = manifest["width"], manifest["height"]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in manifest["chunks"]:
            with Image.open(path / c["file"]) as im:
                rgb[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = np.asarray(im.convert("RGB"))
        return rgb, float(manifest.get("pixel_size_m", 10)), len(manifest["chunks"])
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8), 10.0, 1


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_ok(args) -> int:
    behavior = os.getenv("FAKE_CORE", "ok")
    drop = set(filter(None, os.getenv("FAKE_CORE_DROP", "").split(",")))
    have = engines()
    if args.mode == "opt" and args.kmeans in ("omp", "cuda") and not have[args.kmeans]:
        print(f"landcover: mesin tidak tersedia: --kmeans {args.kmeans}: tidak terhubung pada build ini",
              file=sys.stderr)
        return 3
    device = "serial"
    if args.mode == "opt":
        device = args.kmeans if args.kmeans in ("omp", "cuda") else ("omp" if have["omp"] else "serial")

    rgb, pixel_size, n_chunks = load_image(Path(args.input))
    h, w = rgb.shape[:2]
    px = rgb.reshape(-1, 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:                                   # seperti prepare_out_dir() di landcover.cpp
        if (out / name).is_file():
            (out / name).unlink()

    print(f"[info] dataset '{args.name or Path(args.input).stem}': {w}x{h} piksel, {n_chunks} potongan")
    print("[stage] baca selesai 0.01 s")
    init = init_centroids(px, args.k, args.seed)
    labels, centroids, counts = kmeans_int(px, init, args.iters)
    print("[stage] inisialisasi selesai 0.00 s")
    print("[stage] kmeans selesai 0.02 s")
    if behavior == "utf8":
        print("[info] auto: N = 0,0 juta piksel ≥ ambang 8 juta")
    print("[peringatan] OpenMP dan CUDA tidak terhubung; memakai serial" if device == "serial" and args.mode == "opt"
          else "[info] mesin siap")

    labels2d = labels.reshape(h, w).astype(np.uint8)
    colors = centroids[labels2d].astype(np.uint8)
    Image.fromarray(labels2d).save(out / "labels_preview.png", format="PNG")
    Image.fromarray(colors).save(out / "map_preview.png", format="PNG")
    if args.write_labels:
        labels2d.tofile(out / "labels.u8")
    print("[stage] pewarnaan selesai 0.00 s")
    Image.fromarray(colors).save(out / "map.png", format="PNG")
    print("[stage] tulis selesai 0.01 s")

    timing = {"unit": "ms", "read_ms": 10.0, "init_ms": 1.0, "kmeans_ms": 20.0, "color_ms": 1.0,
              "write_ms": 10.0, "total_ms": 42.0, "h2d_ms": 0.0, "kernel_ms": 0.0, "d2h_ms": 0.0,
              "aux_ms": 3.0,
              "devices": {"read": "serial", "init": "serial", "kmeans": device, "color": "serial",
                          "write": "serial"}}
    meta = {
        "contract_version": 1, "mock": behavior == "mock",
        "dataset": "lain" if behavior == "wrongname" else (args.name or Path(args.input).stem),
        "mode": args.mode, "kmeans_device": device, "kmeans_impl": "ref_kmeans",
        "width": w, "height": h, "pixel_size_m": pixel_size, "k": args.k, "iters": args.iters,
        "seed": args.seed, "threads": {"serial": 1, "omp": args.threads, "cuda": 0}[device],
        "io_threads": 1, "init_centroids": init.tolist(), "preview_width": w, "preview_height": h,
        "chunks": n_chunks, "labels_written": bool(args.write_labels), "host": "fake_core (uji)",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    for name, data in (("timing.json", timing), ("centroids.json", centroids.tolist()),
                       ("counts.json", counts.tolist())):
        if name not in drop:
            write_json(out / name, data)
    if "counts.json" not in drop and behavior == "badjson":
        (out / "counts.json").write_text("{bukan json", encoding="utf-8")
    for name in drop:
        if (out / name).is_file():
            (out / name).unlink()
    if "meta.json" not in drop:
        write_json(out / "meta.json", meta)
    print(f"[info] folder run: {out}")
    return 0


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", line_buffering=True)
    if argv[:1] == ["--info"]:
        e = engines()
        print(json.dumps({"program": "landcover", "contract_version": 1, "openmp": True, "max_threads": 4,
                          "engines": e, "auto_cuda_threshold_pixels": 8000000}))
        return 0

    p = argparse.ArgumentParser(prog="landcover", add_help=False)
    p.add_argument("--mode", required=True, choices=("cpu", "opt"))
    p.add_argument("--kmeans", default="auto", choices=("auto", "omp", "cuda"))
    p.add_argument("--input", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--write-labels", action="store_true")
    args = p.parse_args(argv)                                # argumen salah: argparse keluar dengan kode 2

    behavior = os.getenv("FAKE_CORE", "ok")
    log = os.getenv("FAKE_CORE_LOG")
    if log:
        out = Path(args.out)
        seen = sorted(os.listdir(out)) if out.is_dir() else None
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": argv, "out_files_at_start": seen}) + "\n")

    if behavior == "staleguard":                             # pembersihan di launcher harus sudah terjadi
        out = Path(args.out)
        left = [n for n in RUN_FILES if (out / n).exists()]
        if left:
            print(f"berkas basi masih ada: {left}", file=sys.stderr)
            return 9
        return run_ok(args)
    if behavior == "exit1":
        print("[info] dataset 'x': 1x1 piksel")
        print("landcover: galat: masukan rusak (PNG terpotong)", file=sys.stderr)
        return 1
    if behavior == "exit2":
        print("landcover: argumen salah: --k: harus bilangan bulat\npemakaian: ...", file=sys.stderr)
        return 2
    if behavior == "exit3":
        print("landcover: mesin tidak tersedia: --kmeans cuda: kmeans_cuda tidak terhubung", file=sys.stderr)
        return 3
    if behavior == "exit7":
        return 7
    if behavior == "silent0":
        return 0
    if behavior == "garbage":
        print("\x00\x01 ??? keluaran rusak \x7f " + "A" * 1000)
        print("[stage] baca selesai tidak-angka")
        return 0
    if behavior == "hang":
        print("[info] mulai", flush=True)
        time.sleep(30)
        return 0
    if behavior == "bigerr":
        chunk = "e" * 1023 + "\n"
        for i in range(1100):                                # ~1,1 MB ke stderr, jauh di atas isi pipa
            sys.stderr.write(chunk)
            if i % 100 == 0:
                print(f"[info] langkah {i}")
        print("landcover: galat: akhir stderr panjang", file=sys.stderr)
        return 1
    if behavior == "signal":
        import signal
        os.kill(os.getpid(), signal.SIGKILL)
    return run_ok(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
