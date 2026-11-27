#!/usr/bin/env python3
"""Profiling baseline L0 (CPU serial) per tahap, memakai CLI `build/landcover`. Pemilik: Gavra.

Tiap pengukuran = satu run `landcover --mode cpu` (kmeans_cpu, 1 thread); angka diambil dari
timing.json yang ditulis CLI (aturan waktu sama dengan mode opt: read, init, kmeans, color, write,
total; aux_ms TIDAK masuk total). Run pertama tiap konfigurasi dibuang (pemanasan cache disk),
sisanya dipakai dan dilaporkan sebagai median.

Contoh (jalankan dari akar repo, di WSL, setelah `make landcover`):
  # citra asli (disarankan untuk laporan)
  python3 bench/profile_l0.py --image data/sentinel_a.png --image data/sentinel_b.png --k 8
  # sapuan ukuran citra sintetis
  python3 bench/profile_l0.py --sizes 1000x1000,2000x2000,4000x3000,7680x4320 --k 8
  # sapuan K pada satu ukuran
  python3 bench/profile_l0.py --sizes 4000x3000 --k 4,8,16

Keluaran: baris mentah ke bench/results/l0_profile.csv (kolom sama dengan CSV Wisam) dan tabel
Markdown ke docs/PROFIL_L0_tabel.md (tempel ke docs/PROFIL_L0.md).
Citra sintetis butuh numpy dan pillow (sudah ada di app/requirements.txt bila terpasang).
"""
import argparse, csv, json, os, shutil, statistics, subprocess, sys, tempfile

STAGES = ["read_ms", "init_ms", "kmeans_ms", "color_ms", "write_ms"]
NAMES = {"read_ms": "baca", "init_ms": "inisialisasi", "kmeans_ms": "K-Means",
         "color_ms": "pewarnaan", "write_ms": "tulis"}
COLS = ["mode", "image", "width", "height", "k", "iters", "threads", "read_ms", "init_ms",
        "kmeans_ms", "color_ms", "write_ms", "total_ms", "h2d_ms", "kernel_ms", "d2h_ms"]


def make_landscape(path, w, h, seed=1):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    pal = np.array([[30, 70, 160], [40, 120, 50], [170, 150, 100], [150, 150, 150], [90, 160, 70]], np.int16)
    img = np.empty((h, w, 3), np.uint8)
    for y0 in range(0, h, 512):                      # per blok baris agar hemat memori
        y1 = min(h, y0 + 512)
        y, x = np.mgrid[y0:y1, 0:w]
        reg = ((x // 97) * 7 + (y // 131) * 13 + (x // 311) * (y // 277)) % 5
        blk = pal[reg] + rng.integers(-8, 9, (y1 - y0, w, 3))
        img[y0:y1] = np.clip(blk, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def run_once(exe, png, name, k, iters, seed, out):
    cmd = [exe, "--mode", "cpu", "--input", png, "--name", name, "--k", str(k),
           "--iters", str(iters), "--seed", str(seed), "--threads", "1", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("landcover gagal (kode %d):\n%s" % (r.returncode, r.stderr or r.stdout))
    with open(os.path.join(out, "timing.json")) as f:
        t = json.load(f)
    with open(os.path.join(out, "meta.json")) as f:
        m = json.load(f)
    return t, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landcover", default="build/landcover")
    ap.add_argument("--image", action="append", default=[], help="PNG masukan (boleh berulang)")
    ap.add_argument("--sizes", default="", help="ukuran sintetis, mis. 1000x1000,2000x2000")
    ap.add_argument("--k", default="8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=3, help="pengulangan yang dipakai (di luar 1 run pemanasan)")
    ap.add_argument("--csv", default="bench/results/l0_profile.csv")
    ap.add_argument("--md", default="docs/PROFIL_L0_tabel.md")
    a = ap.parse_args()

    if not os.path.exists(a.landcover):
        sys.exit("%s tidak ada; jalankan `make landcover` dari akar repo (di WSL)." % a.landcover)
    ks = [int(v) for v in a.k.split(",") if v]
    tmp = tempfile.mkdtemp(prefix="profil_l0_")
    jobs = []                                         # (label, png)
    for p in a.image:
        jobs.append((os.path.basename(p), p))
    for s in [v for v in a.sizes.split(",") if v]:
        w, h = (int(x) for x in s.split("x"))
        p = os.path.join(tmp, "sintetis_%dx%d.png" % (w, h))
        print("membuat citra sintetis %dx%d ..." % (w, h), flush=True)
        make_landscape(p, w, h)
        jobs.append(("sintetis_lanskap_%dx%d" % (w, h), p))
    if not jobs:
        sys.exit("beri --image atau --sizes")

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    rows_md = []
    impl = ""
    with open(a.csv, "a", newline="") as fcsv:
        wr = csv.writer(fcsv, lineterminator="\n")
        if new:
            wr.writerow(COLS)
        for label, png in jobs:
            for k in ks:
                out = os.path.join(tmp, "run")
                run_once(a.landcover, png, label, k, a.iters, a.seed, out)       # pemanasan, dibuang
                samples, W, H = [], 0, 0
                for _ in range(a.reps):
                    t, m = run_once(a.landcover, png, label, k, a.iters, a.seed, out)
                    W, H = m.get("width", 0), m.get("height", 0)
                    impl = str(m.get("engine", m.get("impl", impl)))
                    samples.append(t)
                    wr.writerow(["cpu", label, W, H, k, a.iters, 1] +
                                ["%.3f" % t[c] for c in STAGES + ["total_ms", "h2d_ms", "kernel_ms", "d2h_ms"]])
                med = {c: statistics.median(s[c] for s in samples) for c in STAGES + ["total_ms"]}
                top = max(STAGES, key=lambda c: med[c])
                rows_md.append((label, W, H, k, med, top))
                print("%-28s %5dx%-5d K=%-2d | " % (label, W, H, k) +
                      " ".join("%s %.1f" % (NAMES[c], med[c]) for c in STAGES) +
                      " | total %.1f ms | bottleneck: %s (%.1f%%)" %
                      (med["total_ms"], NAMES[top], 100 * med[top] / med["total_ms"]), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("Waktu L0 (kmeans_cpu, 1 thread), median dari %d pengulangan, %d iterasi, ms (persen dari total).\n\n" % (a.reps, a.iters))
        f.write("| Citra | Ukuran | K | Baca | Init | K-Means | Pewarnaan | Tulis | Total | Bottleneck |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for label, W, H, k, med, top in rows_md:
            tot = med["total_ms"]
            cells = " | ".join("%.1f (%.1f%%)" % (med[c], 100 * med[c] / tot) for c in STAGES)
            f.write("| %s | %dx%d | %d | %s | %.1f | %s |\n" % (label, W, H, k, cells, tot, NAMES[top]))
    print("\nbaris mentah: %s\ntabel Markdown: %s" % (a.csv, a.md))


if __name__ == "__main__":
    main()
