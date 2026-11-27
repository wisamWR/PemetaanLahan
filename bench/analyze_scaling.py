#!/usr/bin/env python3
"""Analisis benchmark: waktu per tahap, speedup, efisiensi, strong/weak scaling. Pemilik: Gavra.

Membaca *_summary.csv keluaran bench/run_bench.py (median per konfigurasi) dan menulis laporan
Markdown (+ grafik PNG bila matplotlib ada). TIDAK menjalankan pengukuran dan TIDAK mengisi angka
yang tidak ada di CSV: apa pun yang belum terukur ditulis "belum diukur".

Hanya memakai pustaka standar; matplotlib opsional (tanpa itu, grafik dilewati).

Contoh (dari akar repo):
  python3 bench/analyze_scaling.py --summary bench/results/bench_gpu_summary.csv \\
      --label "Mesin B (i7-13620H + RTX 3050)" --md docs/BENCH_scaling.md --figdir docs/fig
  # beberapa CSV dari mesin yang SAMA digabung dengan mengulang --summary;
  # mesin berbeda: jalankan terpisah dengan --label dan --md berbeda (jangan dicampur).

Definisi (T = jumlah thread, t(.) = median waktu ms):
  Speedup terhadap L0        S   = t_L0 / t_konfigurasi            (citra dan K sama; per tahap atau total)
  Speedup strong scaling     S_T = t_omp_x1 / t_omp_xT
  Efisiensi strong           E_T = S_T / T
  Karp-Flatt (fraksi serial) e_T = (1/S_T - 1/T) / (1 - 1/T), T > 1. e konstan = Amdahl murni;
                             e naik bersama T = ada overhead paralel (sinkronisasi, bandwidth, hyperthread).
  Efisiensi weak             E_w = t_omp_x1(beban dasar) / t_omp_xT(beban dasar x T)   (ideal 1,0)
  Throughput K-Means         ns per piksel per iterasi = kmeans_ms * 1e6 / (piksel * iterasi)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

NA = "belum diukur"
STAGES = ["read_ms", "init_ms", "kmeans_ms", "color_ms", "write_ms"]
STAGE_NAME = {"read_ms": "Baca", "init_ms": "Init", "kmeans_ms": "K-Means", "color_ms": "Warna",
              "write_ms": "Tulis", "total_ms": "Total"}
FLOATS = STAGES + ["total_ms", "h2d_ms", "kernel_ms", "d2h_ms", "warmup_ms", "wall_ms"]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for c in FLOATS:
                    r[c] = f(r.get(c))
                for c in ("width", "height", "k", "iters", "threads"):
                    r[c] = int(float(r[c])) if r.get(c) not in (None, "") else 0
                r["px"] = r["width"] * r["height"]
                r["src"] = os.path.basename(p)
                rows.append(r)
    return rows


def idx(rows):
    """(group, config, image, k) -> baris. Bila ganda, yang terakhir menang."""
    return {(r["group"], r["config"], r["image"], r["k"]): r for r in rows}


def short(img):
    return img.replace("sintetis_lanskap_", "")


def n(v, spec="%.1f"):
    return NA if v is None else spec % v


def ratio(a, b):
    return None if (a is None or b is None or b <= 0) else a / b


def karp_flatt(s, t):
    if s is None or t <= 1 or s <= 0:
        return None
    return (1.0 / s - 1.0 / t) / (1.0 - 1.0 / t)


def table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out) + "\n"


def fits(rows, group):
    return [r for r in rows if r["group"] == group]


def sizes_k(rows, group="strong"):
    """Pasangan (image, k) yang punya baris L0, urut piksel lalu K."""
    seen = {}
    for r in fits(rows, group):
        if r["config"] == "L0":
            seen[(r["image"], r["k"])] = r["px"]
    return sorted(seen, key=lambda x: (x[1], seen[x]))


# ---------------------------------------------------------------------------------------
def sec_per_stage(rows, I):
    o = ["## 1. Waktu per tahap (ms) dan porsi dari total\n",
         "Median dari 5 pengulangan setelah 1 pemanasan. Persen = porsi terhadap Total konfigurasi itu sendiri. "
         "`L1+L2 cuda xT` = K-Means di GPU dengan baca/tulis/warna OpenMP T thread; `L2 omp xT` = K-Means OpenMP "
         "dengan baca/tulis/warna OpenMP T thread.\n"]
    for (img, k) in sizes_k(rows):
        cfgs = [r for r in fits(rows, "strong") if r["image"] == img and r["k"] == k]
        if not cfgs:
            continue
        o.append("\n**%s, K=%d**\n" % (short(img), k))
        body = []
        for r in cfgs:
            tot = r["total_ms"]
            cells = []
            for c in STAGES:
                v = r[c]
                cells.append("%.1f (%.0f%%)" % (v, 100 * v / tot) if tot else "%.1f" % v)
            body.append([r["config"]] + cells + ["%.1f" % tot])
        o.append(table(["Konfigurasi"] + [STAGE_NAME[c] for c in STAGES] + ["Total"], body))
    return "\n".join(o)


def best(rows, group, prefix, img, k, key="total_ms"):
    c = [r for r in rows if r["group"] == group and r["config"].startswith(prefix)
         and r["image"] == img and r["k"] == k and r[key] is not None]
    return min(c, key=lambda r: r[key]) if c else None


def sec_steps(rows, I):
    o = ["## 2. Sebelum dan sesudah tiap langkah optimasi (L0 → L1 → L2 → L1+L2)\n",
         "Speedup terhadap L0 pada citra dan K yang sama. Kolom K-Means memakai `kmeans_ms`; kolom Total memakai "
         "`total_ms` (tanpa pemanasan konteks CUDA, lihat bagian 7). Konfigurasi terbaik dipilih dari Total.\n"]
    body = []
    for (img, k) in sizes_k(rows):
        l0 = I.get(("strong", "L0", img, k))
        l1 = I.get(("strong", "L1 cuda", img, k))
        om = best(rows, "strong", "L2 omp", img, k)
        l12 = best(rows, "strong", "L1+L2 cuda", img, k)
        if not l0:
            continue
        for name, r in (("L0", l0), ("L2 (OpenMP) terbaik", om), ("L1 (CUDA)", l1), ("L1+L2 terbaik", l12)):
            if r is None:
                body.append([short(img), k, name, NA, NA, NA, NA, NA])
                continue
            body.append([short(img), k, r["config"],
                         "%.1f" % r["kmeans_ms"], n(ratio(l0["kmeans_ms"], r["kmeans_ms"]), "%.1fx"),
                         "%.1f" % r["total_ms"], n(ratio(l0["total_ms"], r["total_ms"]), "%.2fx"),
                         "%.1f" % (r["read_ms"] + r["write_ms"] + r["color_ms"] + r["init_ms"])])
    o.append(table(["Citra", "K", "Konfigurasi", "K-Means (ms)", "Speedup K-Means", "Total (ms)", "Speedup Total",
                    "Di luar K-Means (ms)"], body))
    return "\n".join(o)


def sec_strong(rows, I):
    o = ["## 3. Strong scaling OpenMP (ukuran tetap, thread naik)\n",
         "Basis = `L2 omp x1` (bukan L0), supaya overhead OpenMP tidak tercampur ke efisiensi. Dua sudut pandang: "
         "**K-Means saja** (komputasi murni) dan **Total** (termasuk tahap serial seperti baca PNG). Selisih keduanya "
         "adalah bagian yang tidak ikut menskala. `e` = fraksi serial Karp-Flatt.\n"]
    for (img, k) in sizes_k(rows):
        b1 = I.get(("strong", "L2 omp x1", img, k))
        if not b1:
            continue
        pts = sorted([r for r in fits(rows, "strong") if r["image"] == img and r["k"] == k and r["mode"] == "omp"],
                     key=lambda r: r["threads"])
        body = []
        for r in pts:
            T = r["threads"]
            sk, st, sw = (ratio(b1["kmeans_ms"], r["kmeans_ms"]), ratio(b1["total_ms"], r["total_ms"]),
                          ratio(b1["write_ms"], r["write_ms"]))
            body.append([T, "%.1f" % r["kmeans_ms"], n(sk, "%.2fx"), n(sk / T if sk else None, "%.2f"),
                         n(karp_flatt(sk, T), "%.3f") if T > 1 else "-",
                         "%.1f" % r["total_ms"], n(st, "%.2fx"), n(st / T if st else None, "%.2f"),
                         n(sw / T if sw else None, "%.2f")])
        o.append("\n**%s, K=%d**\n" % (short(img), k))
        o.append(table(["T", "K-Means (ms)", "S K-Means", "E K-Means", "e K-Means", "Total (ms)", "S Total",
                        "E Total", "E Tulis"], body))
    return "\n".join(o)


def sec_weak(rows):
    wk = fits(rows, "weak")
    o = ["## 4. Weak scaling OpenMP (beban per thread tetap)\n"]
    if not wk:
        o.append("**%s.** Tidak ada baris `group=weak` pada CSV yang dibaca. Jalankan: "
                 "`python3 bench/run_bench.py --weak 1000x1000 --engines omp --k 8 --tag weak` (lihat "
                 "`docs/BENCH_runbook.md`).\n" % NA)
        return "\n".join(o)
    o.append("Piksel = beban dasar x T. Efisiensi weak = t(T=1) / t(T), ideal 1,0. Dihitung untuk **K-Means saja** "
             "dan **Total**. Total biasanya jatuh karena baca PNG serial ikut membesar bersama citra.\n")
    byk = defaultdict(list)
    for r in wk:
        byk[r["k"]].append(r)
    for k, rs in sorted(byk.items()):
        rs.sort(key=lambda r: r["threads"])
        b1 = next((r for r in rs if r["threads"] == 1), None)
        body = []
        for r in rs:
            ek = ratio(b1["kmeans_ms"], r["kmeans_ms"]) if b1 else None
            et = ratio(b1["total_ms"], r["total_ms"]) if b1 else None
            body.append([r["threads"], "%dx%d" % (r["width"], r["height"]), "%.1f" % (r["px"] / 1e6),
                         "%.1f" % r["kmeans_ms"], n(ek, "%.2f"), "%.1f" % r["total_ms"], n(et, "%.2f")])
        o.append("\n**K=%d**\n" % k)
        o.append(table(["T", "Ukuran", "Piksel (juta)", "K-Means (ms)", "E weak K-Means", "Total (ms)",
                        "E weak Total"], body))
    return "\n".join(o)


def sec_throughput(rows):
    o = ["## 5. Throughput K-Means menurut ukuran citra (analog weak scaling untuk GPU)\n",
         "ns per piksel per iterasi = `kmeans_ms` x 1e6 / (piksel x iterasi). Nilai yang datar terhadap ukuran "
         "berarti waktu tumbuh linear dengan beban; nilai yang turun berarti biaya tetap (peluncuran kernel, "
         "transfer awal) makin terbagi.\n"]
    cfgs = []
    for r in fits(rows, "strong"):
        if r["config"] not in cfgs and (r["config"] in ("L0", "L1 cuda", "L2 omp x1") or r["config"].startswith("L2 omp x")):
            cfgs.append(r["config"])
    ks = sorted({r["k"] for r in fits(rows, "strong")})
    for k in ks:
        imgs = sorted({(r["px"], r["image"]) for r in fits(rows, "strong") if r["k"] == k})
        body = []
        for px, img in imgs:
            line = [short(img), "%.2f" % (px / 1e6)]
            for c in cfgs:
                r = next((x for x in fits(rows, "strong") if x["config"] == c and x["image"] == img and x["k"] == k), None)
                line.append(NA if r is None else "%.2f" % (r["kmeans_ms"] * 1e6 / (r["px"] * r["iters"])))
            body.append(line)
        if body:
            o.append("\n**K=%d**\n" % k)
            o.append(table(["Citra", "Piksel (juta)"] + cfgs, body))
    return "\n".join(o)


def sec_k(rows):
    o = ["## 6. Pengaruh jumlah cluster K\n"]
    by_img = defaultdict(set)
    for r in fits(rows, "strong"):
        by_img[r["image"]].add(r["k"])
    multi = [i for i, ks in by_img.items() if len(ks) > 1]
    if not multi:
        o.append("**%s.** Semua citra pada CSV ini hanya diukur dengan satu nilai K. Jalankan sapuan K "
                 "(`docs/BENCH_runbook.md`, langkah 2).\n" % NA)
        return "\n".join(o)
    o.append("K-Means (ms) per K. Rasio terhadap K terkecil menunjukkan apakah waktu tumbuh linear terhadap K "
             "(OpenMP/CPU diharapkan mendekati linear; CUDA lebih datar).\n")
    for img in sorted(multi):
        ks = sorted(by_img[img])
        cfgs = []
        for r in fits(rows, "strong"):
            if r["image"] == img and r["config"] not in cfgs:
                cfgs.append(r["config"])
        body = []
        for c in cfgs:
            line = [c]
            base = None
            for k in ks:
                r = next((x for x in fits(rows, "strong") if x["config"] == c and x["image"] == img and x["k"] == k), None)
                if r is None:
                    line.append(NA)
                    continue
                base = base or r["kmeans_ms"]
                line.append("%.1f (%.2fx)" % (r["kmeans_ms"], r["kmeans_ms"] / base))
            body.append(line)
        o.append("\n**%s**\n" % short(img))
        o.append(table(["Konfigurasi"] + ["K=%d" % k for k in ks], body))
    return "\n".join(o)


def sec_amdahl(rows, I):
    o = ["## 7. Bottleneck setelah optimasi dan batas Amdahl\n",
         "Setelah K-Means dipercepat, tahap yang tidak ikut dipercepat menentukan batas. Kolom `Batas` = "
         "total konfigurasi / (total - K-Means) hanya sebagai ilustrasi: speedup tambahan paling besar yang masih "
         "mungkin bila K-Means menjadi 0 ms dan tahap lain tetap.\n"]
    body = []
    for (img, k) in sizes_k(rows):
        for name, r in (("L1+L2 terbaik", best(rows, "strong", "L1+L2 cuda", img, k)),
                        ("L2 omp terbaik", best(rows, "strong", "L2 omp", img, k))):
            if r is None or not r["total_ms"]:
                continue
            rest = r["total_ms"] - r["kmeans_ms"]
            big = max(("read_ms", "write_ms", "color_ms", "init_ms"), key=lambda c: r[c])
            body.append([short(img), k, r["config"], "%.0f%%" % (100 * r["kmeans_ms"] / r["total_ms"]),
                         "%s (%.0f%%)" % (STAGE_NAME[big], 100 * r[big] / r["total_ms"]),
                         "%.2fx" % (r["total_ms"] / rest) if rest > 0 else NA])
    o.append(table(["Citra", "K", "Konfigurasi", "Porsi K-Means", "Tahap terbesar di luar K-Means", "Batas"], body))
    hasw = any(r.get("warmup_ms") for r in rows)
    o.append("\n**Pemanasan konteks CUDA.** `total_ms` tidak memuat inisialisasi konteks CUDA (sekitar 250 ms pada "
             "pengukuran 22 Nov di kontrak U8; angka terbaru ada di `warmup_ms`), tetapi aplikasi menjalankan `landcover` sebagai proses baru tiap run, jadi pengguna "
             "membayarnya. %s\n" % ("Kolom `warmup_ms` dan `wall_ms` ada di CSV ini; lihat `docs/BENCH_tabel.md` "
                                    "untuk perbandingan Proses." if hasw else
                                    "CSV ini tidak punya kolom `warmup_ms`/`wall_ms`, jadi pengaruhnya %s di sini." % NA))
    return "\n".join(o)


def sec_variants(rows):
    v = fits(rows, "cuda_variants")
    o = ["## 8. Varian CUDA (T3 simple, T4 shared, T5 streams), tahap K-Means saja\n"]
    if not v:
        o.append("**%s** (tidak ada baris `cuda_variants`).\n" % NA)
        return "\n".join(o)
    o.append("Dari `tests/bench_cuda` pada citra acak, bukan lewat `landcover`; sebab itu tidak mencakup baca/tulis. "
             "`landcover` belum bisa memanggil T3/T5, jadi L3 belum bisa diukur sebagai pipeline utuh.\n")
    keys = sorted({(r["px"], r["k"]) for r in v})
    cfgs = sorted({r["config"] for r in v})
    body = []
    for px, k in keys:
        line = ["%dx%d" % next((r["width"], r["height"]) for r in v if r["px"] == px), k]
        vals = {}
        for c in cfgs:
            r = next((x for x in v if x["config"] == c and x["px"] == px and x["k"] == k), None)
            vals[c] = r["kmeans_ms"] if r else None
            line.append(n(vals[c]))
        simple = vals.get("bench_cuda simple")
        others = [c for c in cfgs if c != "bench_cuda simple"]
        for c in others:
            line.append(n(ratio(simple, vals[c]), "%.1fx"))
        body.append(line)
    o.append(table(["Ukuran", "K"] + [c.replace("bench_cuda ", "") + " (ms)" for c in cfgs] +
                   ["x vs simple: " + c.replace("bench_cuda ", "") for c in others], body))
    return "\n".join(o)


# ---------------------------------------------------------------------------------------
def figures(rows, I, figdir, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib tidak ada; grafik dilewati.")
        return []
    os.makedirs(figdir, exist_ok=True)
    made = []
    pairs = sizes_k(rows)
    if not pairs:
        return made
    big = max(pairs, key=lambda p: (I[("strong", "L0", p[0], p[1])]["px"], p[1] == 8))
    img, k = big

    cfgs = [r for r in fits(rows, "strong") if r["image"] == img and r["k"] == k]
    pick = [r for r in cfgs if r["config"] in ("L0", "L1 cuda") or r["config"].endswith("x1")]
    for pre in ("L2 omp", "L1+L2 cuda"):
        b = best(rows, "strong", pre, img, k)
        if b:
            pick.append(b)
    seen, order = set(), []
    for r in pick:
        if r["config"] not in seen:
            seen.add(r["config"])
            order.append(r)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = [0] * len(order)
    for c in STAGES:
        vals = [r[c] for r in order]
        ax.barh([r["config"] for r in order], vals, left=left, label=STAGE_NAME[c])
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("ms (median)")
    ax.set_title("Waktu per tahap, %s K=%d\n%s" % (short(img), k, label), fontsize=9)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "tahap_%s_k%d.png" % (short(img), k))
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for (im, kk) in pairs:
        if kk != k:
            continue
        b1 = I.get(("strong", "L2 omp x1", im, kk))
        pts = sorted([r for r in fits(rows, "strong") if r["image"] == im and r["k"] == kk and r["mode"] == "omp"],
                     key=lambda r: r["threads"])
        if not b1 or len(pts) < 2:
            continue
        T = [r["threads"] for r in pts]
        axs[0].plot(T, [b1["kmeans_ms"] / r["kmeans_ms"] for r in pts], "o-", label=short(im))
        axs[1].plot(T, [b1["kmeans_ms"] / r["kmeans_ms"] / r["threads"] for r in pts], "o-", label=short(im))
    mx = max([r["threads"] for r in fits(rows, "strong") if r["mode"] == "omp"] or [1])
    axs[0].plot([1, mx], [1, mx], "k--", lw=1, label="ideal")
    axs[1].axhline(1.0, color="k", ls="--", lw=1)
    axs[0].set_title("Speedup K-Means (strong), K=%d" % k, fontsize=9)
    axs[1].set_title("Efisiensi K-Means (strong), K=%d" % k, fontsize=9)
    for a in axs:
        a.set_xlabel("thread")
        a.set_xscale("log", base=2)
        a.grid(alpha=.3)
    axs[0].legend(fontsize=7)
    fig.suptitle(label, fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "strong_kmeans_k%d.png" % k)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    wk = fits(rows, "weak")
    if wk:
        fig, ax = plt.subplots(figsize=(5, 4))
        for kk in sorted({r["k"] for r in wk}):
            rs = sorted([r for r in wk if r["k"] == kk], key=lambda r: r["threads"])
            b1 = next((r for r in rs if r["threads"] == 1), None)
            if b1:
                ax.plot([r["threads"] for r in rs], [b1["kmeans_ms"] / r["kmeans_ms"] for r in rs], "o-", label="K=%d" % kk)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("thread (piksel naik sebanding)")
        ax.set_ylabel("efisiensi weak K-Means")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(figdir, "weak_kmeans.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="append", required=True, help="*_summary.csv dari run_bench.py (boleh berulang)")
    ap.add_argument("--label", default="", help="nama mesin, mis. 'Mesin B (i7-13620H + RTX 3050)'")
    ap.add_argument("--md", default="docs/BENCH_scaling.md")
    ap.add_argument("--figdir", default="docs/fig")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()

    rows = load(a.summary)
    bad = [r for r in rows if r.get("identik") == "BEDA"]
    if bad:
        sys.exit("ADA hasil bertanda BEDA di CSV; perbaiki dulu, jangan dianalisis: %s" %
                 ", ".join("%s|%s|K=%s" % (r["config"], short(r["image"]), r["k"]) for r in bad[:5]))
    I = idx(rows)

    head = ["# Analisis scaling benchmark\n",
            "Mesin: **%s**. Sumber: %s. Dibuat oleh `bench/analyze_scaling.py` dari CSV hasil ukur; "
            "tidak ada angka yang diisi manual. Semua hasil sudah ditandai identik dengan acuan oleh `run_bench.py` "
            "(centroid dan counts); verifikasi label penuh adalah tugas modul verifikasi.\n" %
            (a.label or "(isi dengan --label)", ", ".join("`%s`" % os.path.basename(p) for p in a.summary)),
            "Angka dari mesin berbeda tidak boleh dicampur dalam satu tabel.\n"]
    parts = head + [sec_per_stage(rows, I), sec_steps(rows, I), sec_strong(rows, I), sec_weak(rows),
                    sec_throughput(rows), sec_k(rows), sec_amdahl(rows, I), sec_variants(rows)]
    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    figs = [] if a.no_figs else figures(rows, I, a.figdir, a.label)
    if figs:
        parts.append("## Grafik\n\n" + "\n".join("![%s](%s)" % (os.path.basename(p), os.path.relpath(p, os.path.dirname(a.md) or ".")) for p in figs) + "\n")
    with open(a.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print("tulis %s (%d baris CSV dibaca)%s" % (a.md, len(rows), ", grafik: %s" % ", ".join(figs) if figs else ""))


if __name__ == "__main__":
    main()
