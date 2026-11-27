#!/usr/bin/env python3
"""Skrip benchmark L0 -> L1 -> L2 -> L1+L2 memakai CLI `build/landcover`. Pemilik: Gavra.

Satu konfigurasi = (citra, K, mesin, jumlah thread). Tiap konfigurasi dijalankan 1 kali untuk
pemanasan (dibuang), lalu --reps kali (kontrak butir 17: minimal 5). Angka waktu diambil dari
timing.json yang ditulis landcover (aturan sama dengan kontrak U8); yang dilaporkan adalah
MEDIAN per kolom. Angka yang tidak bisa diukur ditulis "belum diukur", tidak pernah diisi
perkiraan.

Konfigurasi yang dijalankan (pilih lewat --engines):
  L0              landcover --mode cpu                      (kmeans_cpu, 1 thread, baca/tulis serial)
  L2 omp xT       landcover --mode opt --kmeans omp -T      (K-Means OpenMP + baca/tulis/warna paralel)
  L1 cuda         landcover --mode opt --kmeans cuda -1     (kmeans_cuda, baca/tulis serial)
  L1+L2 cuda xT   landcover --mode opt --kmeans cuda -T     (kmeans_cuda + baca/tulis/warna paralel)
Varian CUDA simple dan streams (L3) TIDAK bisa dipanggil lewat landcover; keduanya diukur lewat
tests/bench_cuda (opsi --cuda-variants), pada citra sintetis acak, hanya tahap K-Means.

Keamanan hasil: tiap run dibandingkan dengan acuan (run pertama pada citra dan K yang sama,
biasanya L0) lewat centroids.json dan counts.json. Hasil yang beda ditandai "BEDA" dan tidak
boleh masuk laporan. Ini pemeriksaan cepat; verifikasi label penuh tetap tugas modul verifikasi.

Contoh (dari akar repo, di WSL, setelah `make landcover`; tambah `make build/bench_cuda` untuk --cuda-variants):
  # rencana saja, tanpa menjalankan
  python3 bench/run_bench.py --sizes 1000x1000,4000x3000 --k 8 --dry-run
  # strong scaling dan perbandingan mesin pada citra sintetis
  python3 bench/run_bench.py --sizes 1000x1000,2000x2000,4000x3000,7680x4320 --k 8
  # citra asli, beberapa K, jumlah thread tertentu
  python3 bench/run_bench.py --image data/sentinel_a.png --k 4,8,16 --threads 1,2,4,8
  # weak scaling (jumlah piksel naik sebanding dengan thread)
  python3 bench/run_bench.py --weak 1000x1000 --engines omp --k 8
  # varian CUDA T3/T4/T5 pada citra sintetis
  python3 bench/run_bench.py --sizes 7680x4320 --engines cuda --cuda-variants simple,shared,streams

Keluaran (--outdir, --tag):
  <tag>_raw.csv      satu baris per pengulangan, 16 kolom persis kontrak butir 14
  <tag>_summary.csv  median per konfigurasi + config, io_threads, speedup, efisiensi, identik
  --md               tabel Markdown (ringkasan, pemenang omp vs cuda per citra, yang dilewati)
Kolom tambahan di summary: warmup_ms (init konteks CUDA, di luar total_ms) dan wall_ms (waktu dinding
satu proses landcover). Untuk ambang mode otomatis pakai wall_ms: aplikasi memanggil landcover sebagai
proses baru tiap run, jadi tiap run CUDA membayar warmup.
"""
import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_l0 import COLS, make_landscape  # noqa: E402  (kolom kontrak dan pembuat citra sintetis yang sama)

STAGES = ["read_ms", "init_ms", "kmeans_ms", "color_ms", "write_ms"]
TIME_COLS = STAGES + ["total_ms", "h2d_ms", "kernel_ms", "d2h_ms"]
NA = "belum diukur"


# ----------------------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------------------
class Config:
    def __init__(self, label, mode, args, io_threads):
        self.label = label          # nama di laporan
        self.mode = mode            # kolom `mode` kontrak: cpu | omp | cuda
        self.args = args            # argumen landcover (tanpa --input/--out/--k/...)
        self.io_threads = io_threads
        self.engine = {"cpu": "cpu", "omp": "omp", "cuda": "cuda"}[mode]


def make_configs(engines, threads):
    cfgs = []
    if "cpu" in engines:
        cfgs.append(Config("L0", "cpu", ["--mode", "cpu", "--threads", "1"], 1))
    if "omp" in engines:
        for t in threads:
            cfgs.append(Config("L2 omp x%d" % t, "omp",
                               ["--mode", "opt", "--kmeans", "omp", "--threads", str(t)], t))
    if "cuda" in engines:
        cfgs.append(Config("L1 cuda", "cuda",
                           ["--mode", "opt", "--kmeans", "cuda", "--threads", "1"], 1))
        for t in threads:
            if t > 1:
                cfgs.append(Config("L1+L2 cuda x%d" % t, "cuda",
                                   ["--mode", "opt", "--kmeans", "cuda", "--threads", str(t)], t))
    return cfgs


def default_threads(max_threads):
    ts, t = set(), 1
    while t <= max_threads:
        ts.add(t)
        t *= 2
    ts.add(max_threads)
    return sorted(ts)


def parse_size(s):
    try:
        w, h = (int(x) for x in s.lower().split("x"))
        if w < 1 or h < 1:
            raise ValueError
        return w, h
    except ValueError:
        sys.exit("ukuran harus berbentuk LEBARxTINGGI, ditemukan '%s'" % s)


# ----------------------------------------------------------------------------------------
# Menjalankan landcover
# ----------------------------------------------------------------------------------------
def landcover_info(exe):
    r = subprocess.run([exe, "--info"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("`%s --info` gagal:\n%s" % (exe, r.stderr or r.stdout))
    return json.loads(r.stdout)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_once(exe, cfg, png, name, k, iters, seed, out, timeout):
    cmd = [exe] + cfg.args + ["--input", png, "--name", name, "--k", str(k),
                              "--iters", str(iters), "--seed", str(seed), "--out", out]
    t_proc = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall_ms = (time.perf_counter() - t_proc) * 1000.0                # seluruh proses: start, warmup, semua tahap
    if r.returncode != 0:
        raise RuntimeError("kode %d: %s" % (r.returncode, (r.stderr or r.stdout).strip()[-300:]))
    t = read_json(os.path.join(out, "timing.json"))
    t["wall_ms"] = wall_ms
    m = read_json(os.path.join(out, "meta.json"))
    sig = json.dumps([read_json(os.path.join(out, "centroids.json")),
                      read_json(os.path.join(out, "counts.json"))])
    return t, m, sig


def median(vals):
    return statistics.median(vals)


# ----------------------------------------------------------------------------------------
# Pengukuran
# ----------------------------------------------------------------------------------------
def measure(a, exe, cfg, png, label, k, ref, raw_w, skipped, group):
    """Satu konfigurasi: pemanasan + reps. Kembalikan dict hasil, atau None bila gagal."""
    out = os.path.join(a.tmp, "run")
    tag = "%s | %s | K=%d" % (cfg.label, label, k)
    try:
        run_once(exe, cfg, png, label, k, a.iters, a.seed, out, a.timeout)          # pemanasan, dibuang
        samples, sigs, W, H, thr = [], [], 0, 0, 0
        for _ in range(a.reps):
            t, m, sig = run_once(exe, cfg, png, label, k, a.iters, a.seed, out, a.timeout)
            samples.append(t)
            sigs.append(sig)
            W, H, thr = m.get("width", 0), m.get("height", 0), m.get("threads", 0)
    except (RuntimeError, subprocess.TimeoutExpired, OSError, KeyError, ValueError) as e:
        skipped.append((tag, "gagal: %s" % e))
        print("  !! %s dilewati: %s" % (tag, e), flush=True)
        return None

    key = (label, k)
    if key not in ref:
        ref[key] = (cfg.label, sigs[0])
    same = all(s == ref[key][1] for s in sigs)
    for t in samples:
        raw_w.writerow([cfg.mode, label, W, H, k, a.iters, thr] + ["%.3f" % t.get(c, 0.0) for c in TIME_COLS])
    med = {c: median(s.get(c, 0.0) for s in samples) for c in TIME_COLS}
    med["warmup_ms"] = median(s.get("warmup_ms", 0.0) for s in samples)   # init konteks CUDA, di luar total_ms
    med["wall_ms"] = median(s["wall_ms"] for s in samples)                # waktu dinding satu proses landcover
    res = dict(group=group, config=cfg.label, mode=cfg.mode, image=label, width=W, height=H, k=k,
               iters=a.iters, threads=thr, io_threads=cfg.io_threads, med=med, reps=a.reps,
               identik="ya" if same else "BEDA", acuan=ref[key][0], speedup=None, efisiensi=None)
    print("  %-16s %-28s %5dx%-5d K=%-2d total %9.1f ms | kmeans %9.1f | %s" %
          (cfg.label, label, W, H, k, med["total_ms"], med["kmeans_ms"], res["identik"]), flush=True)
    if not same:
        print("  !! HASIL BEDA dari acuan (%s): jangan dipakai di laporan, laporkan ke tim." % ref[key][0], flush=True)
    return res


def run_cuda_variants(a, sizes, ks, raw_w, results, skipped):
    """Varian CUDA lewat tests/bench_cuda (citra sintetis acak, hanya tahap K-Means)."""
    if not os.path.exists(a.bench_cuda):
        skipped.append(("varian CUDA (%s)" % a.cuda_variants,
                        "%s tidak ada; jalankan `make build/bench_cuda` di WSL" % a.bench_cuda))
        print("!! %s tidak ada; varian CUDA dilewati." % a.bench_cuda)
        return
    for v in [x for x in a.cuda_variants.split(",") if x]:
        for (w, h) in sizes:
            for k in ks:
                tmp_csv = os.path.join(a.tmp, "bench_cuda_%s.csv" % v)
                if os.path.exists(tmp_csv):
                    os.remove(tmp_csv)
                cmd = [a.bench_cuda, "--variant", v, "--width", str(w), "--height", str(h), "--k", str(k),
                       "--iters", str(a.iters), "--reps", str(a.reps), "--seed", str(a.seed), "--out", tmp_csv]
                tag = "bench_cuda %s | %dx%d | K=%d" % (v, w, h, k)
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                    if r.returncode != 0:
                        raise RuntimeError("kode %d: %s" % (r.returncode, r.stderr.strip()[-200:]))
                    with open(tmp_csv, newline="") as f:
                        row = list(csv.DictReader(f))[-1]
                except (RuntimeError, subprocess.TimeoutExpired, OSError, IndexError) as e:
                    skipped.append((tag, "gagal: %s" % e))
                    print("  !! %s dilewati: %s" % (tag, e), flush=True)
                    continue
                raw_w.writerow([row[c] for c in COLS])             # baris ini sudah median bench_cuda
                med = {c: float(row[c]) for c in TIME_COLS}
                results.append(dict(group="cuda_variants", config="bench_cuda " + v, mode=row["mode"],
                                    image=row["image"], width=w, height=h, k=k, iters=a.iters, threads=0,
                                    io_threads=0, med=med, reps=a.reps, identik="tidak dicek",
                                    acuan="-", speedup=None, efisiensi=None))
                print("  %-20s %5dx%-5d K=%-2d wall %9.1f ms | kmeans %9.1f" %
                      ("bench_cuda " + v, w, h, k, med["total_ms"], med["kmeans_ms"]), flush=True)


# ----------------------------------------------------------------------------------------
# Speedup dan efisiensi
# ----------------------------------------------------------------------------------------
def compute_scaling(results):
    idx = {(r["group"], r["config"], r["image"], r["k"]): r for r in results}
    for r in results:
        if r["group"] == "cuda_variants":
            continue
        base = idx.get((r["group"], "L0", r["image"], r["k"]))
        if base and r["med"]["total_ms"] > 0:
            r["speedup"] = base["med"]["total_ms"] / r["med"]["total_ms"]            # terhadap L0, total
        if r["mode"] == "omp" and r["group"] == "strong":                           # strong scaling
            b1 = idx.get(("strong", "L2 omp x1", r["image"], r["k"]))
            if b1 and r["med"]["total_ms"] > 0:
                r["efisiensi"] = b1["med"]["total_ms"] / r["med"]["total_ms"] / r["threads"]
        if r["group"] == "weak":                                                    # weak scaling
            b1 = next((x for x in results if x["group"] == "weak" and x["threads"] == 1 and x["k"] == r["k"]), None)
            if b1 and r["med"]["total_ms"] > 0:
                r["efisiensi"] = b1["med"]["total_ms"] / r["med"]["total_ms"]


def fnum(v, spec="%.1f"):
    return NA if v is None else spec % v


def write_outputs(a, results, skipped, raw_path, sum_path):
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "group"] + COLS + ["io_threads", "reps", "speedup_total_vs_L0",
                                                  "efisiensi", "identik", "acuan", "warmup_ms",
                                                  "wall_ms"])
        for r in results:
            w.writerow([r["config"], r["group"], r["mode"], r["image"], r["width"], r["height"], r["k"],
                        r["iters"], r["threads"]] + ["%.3f" % r["med"][c] for c in TIME_COLS] +
                       [r["io_threads"], r["reps"], fnum(r["speedup"], "%.3f"), fnum(r["efisiensi"], "%.3f"),
                        r["identik"], r["acuan"], "%.3f" % r["med"].get("warmup_ms", 0.0),
                        fnum(r["med"].get("wall_ms"), "%.3f")])
    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("Benchmark landcover: median dari %d pengulangan (setelah 1 pemanasan), %d iterasi, seed %d, ms.\n"
                "Speedup = total L0 / total konfigurasi (citra dan K sama). Efisiensi strong = speedup omp xT "
                "terhadap omp x1, dibagi T; efisiensi weak = waktu omp x1 / waktu omp xT pada beban per thread tetap. "
                "\"%s\" = tidak terukur. Pemeriksaan identik: centroid dan counts sama dengan acuan.\n"
                "Warmup = inisialisasi konteks CUDA: TIDAK masuk Total, tetapi dibayar di setiap proses landcover "
                "(aplikasi memanggil landcover sebagai proses baru tiap run). Proses = waktu dinding seluruh proses.\n\n"
                % (a.reps, a.iters, a.seed, NA))
        f.write("| Konfigurasi | Citra | Ukuran | K | Baca | Init | K-Means | Warna | Tulis | Total | Warmup | Proses | Speedup | Efisiensi | Identik |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            m = r["med"]
            f.write("| %s | %s | %dx%d | %d | %s | %.1f | %.1f | %s | %s | %s | %s |\n" % (
                r["config"], r["image"], r["width"], r["height"], r["k"],
                " | ".join("%.1f" % m[c] for c in STAGES), m["total_ms"], m.get("warmup_ms", 0.0),
                fnum(m.get("wall_ms")), fnum(r["speedup"], "%.2fx"), fnum(r["efisiensi"], "%.2f"), r["identik"]))
        # Pemenang omp vs cuda per (citra, K): bahan ambang mode otomatis.
        groups = {}
        for r in results:
            if r["group"] == "strong" and r["mode"] in ("omp", "cuda"):
                groups.setdefault((r["image"], r["width"] * r["height"], r["k"]), []).append(r)
        rows = []
        for (img, px, k), rs in sorted(groups.items(), key=lambda x: (x[0][2], x[0][1])):
            o = [r for r in rs if r["mode"] == "omp"]
            c = [r for r in rs if r["mode"] == "cuda"]
            if o and c:
                bo = min(o, key=lambda r: r["med"]["total_ms"])
                bc = min(c, key=lambda r: r["med"]["total_ms"])
                win = "cuda" if bc["med"]["total_ms"] < bo["med"]["total_ms"] else "omp"
                # kriteria kedua: waktu dinding proses (termasuk warmup CUDA); tanpa wall_ms -> total + warmup
                def wall(r):
                    return r["med"].get("wall_ms") or (r["med"]["total_ms"] + r["med"].get("warmup_ms", 0.0))
                wo = min(o, key=wall)
                wc = min(c, key=wall)
                win_w = "cuda" if wall(wc) < wall(wo) else "omp"
                rows.append((img, px, k, bo, bc, win, wo, wc, win_w, wall))
        if rows:
            f.write("\n### Pemenang omp vs cuda (konfigurasi terbaik tiap mesin)\n\n")
            f.write("Dasar penentuan ambang mode otomatis (`kAutoCudaThresholdPixels` di landcover.cpp). "
                    "Dua kriteria: Total (tanpa warmup CUDA) dan Proses (waktu dinding, dengan warmup CUDA). "
                    "Yang dirasakan pengguna adalah Proses.\n\n")
            f.write("| Citra | Piksel (juta) | K | omp: Total | cuda: Total | Menang (Total) | omp: Proses | cuda: Proses | Menang (Proses) |\n"
                    "|---|---|---|---|---|---|---|---|---|\n")
            for img, px, k, bo, bc, win, wo, wc, win_w, wall in rows:
                f.write("| %s | %.2f | %d | %.1f (%s) | %.1f (%s) | %s | %.1f (%s) | %.1f (%s) | %s |\n" % (
                    img, px / 1e6, k, bo["med"]["total_ms"], bo["config"], bc["med"]["total_ms"], bc["config"], win,
                    wall(wo), wo["config"], wall(wc), wc["config"], win_w))
        if skipped:
            f.write("\n### Dilewati (%s)\n\n" % NA)
            for tag, why in skipped:
                f.write("- %s: %s\n" % (tag, why))
        bad = [r for r in results if r["identik"] == "BEDA"]
        if bad:
            f.write("\n### PERINGATAN: hasil tidak identik dengan acuan\n\n")
            for r in bad:
                f.write("- %s | %s | K=%d (acuan: %s)\n" % (r["config"], r["image"], r["k"], r["acuan"]))


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landcover", default="build/landcover")
    ap.add_argument("--image", action="append", default=[], help="PNG masukan (boleh berulang)")
    ap.add_argument("--sizes", default="", help="ukuran sintetis, mis. 1000x1000,4000x3000")
    ap.add_argument("--k", default="8", help="daftar K, mis. 4,8,16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=5, help="pengulangan terukur per konfigurasi (kontrak: minimal 5)")
    ap.add_argument("--engines", default="cpu,omp,cuda", help="subset dari cpu,omp,cuda")
    ap.add_argument("--threads", default="auto", help="daftar thread, mis. 1,2,4,8; auto = pangkat 2 sampai maks mesin")
    ap.add_argument("--weak", default="", help="ukuran dasar weak scaling, mis. 1000x1000 (lebar dikali T)")
    ap.add_argument("--cuda-variants", default="", help="subset dari simple,shared,streams (lewat tests/bench_cuda)")
    ap.add_argument("--bench-cuda", default="build/bench_cuda")
    ap.add_argument("--outdir", default="bench/results")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--md", default="docs/BENCH_tabel.md")
    ap.add_argument("--timeout", type=int, default=900, help="batas detik per run landcover")
    ap.add_argument("--dry-run", action="store_true", help="cetak rencana lalu keluar")
    a = ap.parse_args()

    if a.reps < 1:
        sys.exit("--reps harus >= 1")
    if a.reps < 5:
        print("[peringatan] --reps %d < 5: tidak memenuhi kontrak butir 17, jangan dipakai untuk laporan." % a.reps)
    engines = [e for e in a.engines.split(",") if e]
    bad = [e for e in engines if e not in ("cpu", "omp", "cuda")]
    if bad:
        sys.exit("--engines tidak dikenal: %s" % ",".join(bad))
    ks = [int(v) for v in a.k.split(",") if v]
    sizes = [parse_size(s) for s in a.sizes.split(",") if s]
    if not (a.image or sizes or a.weak):
        sys.exit("beri --image, --sizes, atau --weak")

    if not os.path.exists(a.landcover):
        sys.exit("%s tidak ada; jalankan `make landcover` dari akar repo (di WSL)." % a.landcover)
    info = landcover_info(a.landcover)
    avail = {"cpu": bool(info["engines"].get("cpu")), "omp": bool(info["engines"].get("omp")),
             "cuda": bool(info["engines"].get("cuda"))}
    maxt = int(info.get("max_threads", 1))
    threads = default_threads(maxt) if a.threads == "auto" else sorted({int(v) for v in a.threads.split(",") if v})
    if threads and max(threads) > maxt:
        print("[peringatan] --threads melebihi max_threads mesin (%d): hasilnya oversubscription." % maxt)

    skipped = []
    use_engines = []
    for e in engines:
        if avail[e]:
            use_engines.append(e)
        else:
            skipped.append(("mesin %s" % e, "tidak terhubung pada build ini (landcover --info)"))
            print("[peringatan] mesin %s tidak tersedia pada build ini; dilewati (%s)." % (e, NA))
    cfgs = make_configs(use_engines, threads)

    n_img = len(a.image) + len(sizes)
    runs = n_img * len(ks) * len(cfgs) * (a.reps + 1)
    if a.weak:
        runs += len(threads) * len(ks) * (a.reps + 1) * (1 if "omp" in use_engines else 0)
    print("mesin: %s | thread: %s | K: %s | reps: %d" % (",".join(use_engines), threads, ks, a.reps))
    print("jumlah run landcover: %d (termasuk pemanasan)" % runs)
    if a.dry_run:
        for c in cfgs:
            print("  config %-16s landcover %s" % (c.label, " ".join(c.args)))
        if a.cuda_variants:
            print("  varian CUDA lewat %s: %s" % (a.bench_cuda, a.cuda_variants))
        return

    os.makedirs(a.outdir, exist_ok=True)
    raw_path = os.path.join(a.outdir, "%s_raw.csv" % a.tag)
    sum_path = os.path.join(a.outdir, "%s_summary.csv" % a.tag)
    a.tmp = tempfile.mkdtemp(prefix="bench_")
    results, ref = [], {}
    try:
        with open(raw_path, "w", newline="") as fraw:
            raw_w = csv.writer(fraw, lineterminator="\n")
            raw_w.writerow(COLS)
            jobs = [("strong", os.path.basename(p), p) for p in a.image]
            for (w, h) in sizes:
                p = os.path.join(a.tmp, "sintetis_%dx%d.png" % (w, h))
                print("membuat citra sintetis %dx%d ..." % (w, h), flush=True)
                make_landscape(p, w, h)
                jobs.append(("strong", "sintetis_lanskap_%dx%d" % (w, h), p))
            t_start = time.perf_counter()
            for group, label, png in jobs:
                for k in ks:
                    for cfg in cfgs:
                        r = measure(a, a.landcover, cfg, png, label, k, ref, raw_w, skipped, group)
                        if r:
                            results.append(r)
            if a.weak and "omp" in use_engines:                       # weak scaling: piksel = dasar x T
                bw, bh = parse_size(a.weak)
                for t in threads:
                    p = os.path.join(a.tmp, "weak_%d.png" % t)
                    print("membuat citra weak %dx%d (T=%d) ..." % (bw * t, bh, t), flush=True)
                    make_landscape(p, bw * t, bh)
                    cfg = Config("L2 omp x%d" % t, "omp", ["--mode", "opt", "--kmeans", "omp", "--threads", str(t)], t)
                    for k in ks:
                        r = measure(a, a.landcover, cfg, p, "weak_%dx%d_T%d" % (bw * t, bh, t), k, ref, raw_w, skipped, "weak")
                        if r:
                            results.append(r)
                    os.remove(p)
            if a.cuda_variants:
                run_cuda_variants(a, sizes, ks, raw_w, results, skipped)
            print("selesai dalam %.1f menit" % ((time.perf_counter() - t_start) / 60.0))
    finally:
        shutil.rmtree(a.tmp, ignore_errors=True)

    compute_scaling(results)
    write_outputs(a, results, skipped, raw_path, sum_path)
    print("\nbaris mentah : %s\nringkasan CSV: %s\ntabel Markdown: %s" % (raw_path, sum_path, a.md))
    if any(r["identik"] == "BEDA" for r in results):
        sys.exit(3)


if __name__ == "__main__":
    main()