#!/usr/bin/env python3
"""Membuat tabel Bab V (laporan/tabel/*.tex) dan angka untuk teks (laporan/tabel/angka.tex).

Jalankan dari akar repo (pustaka standar saja):
    .venv\\Scripts\\python.exe laporan/skrip/tabel_ke_tex.py

Sumber (semua angka kinerja dari Mesin B = i7-13620H + RTX 3050 6 GB Laptop, kecuali yang ditandai):
  - bench/results/bench_gpu_summary.csv, run_20261127_1634_{ksweep,weak,cudavar}_summary.csv
    dibaca dengan bench/analyze_scaling.load() dalam urutan yang sama dengan
    docs/BENCH_scaling_mesin_wisam.md, sehingga aturan "pada baris ganda, run 16:34 yang dipakai" identik.
  - bench/results/ambang_summary.csv dibaca dengan bench/fit_auto_threshold.load()/score().
  - docs/PROFIL_L0_tabel.md (Mesin A, i7-1065G7): HANYA tabel pembanding berlabel.
  - laporan/hasil_s2/ (Mesin C, uji fungsional Sentinel-2): jumlah piksel, bukan waktu.

Tidak ada angka yang diketik tangan: semua sel dan semua makro di angka.tex dihitung dari berkas di atas.
Cek silang terhadap docs/BENCH_scaling_mesin_wisam.md, docs/AMBANG_AUTO.md, dan PROGRESS Sesi U
dilakukan dengan assert; skrip berhenti bila tidak cocok. Keluaran deterministik (tanpa cap waktu).
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import analyze_scaling as az  # noqa: E402
import fit_auto_threshold as fat  # noqa: E402

RES = os.path.join(ROOT, "bench", "results")
OUT = os.path.join(ROOT, "laporan", "tabel")
DATA_S2 = os.path.join(ROOT, "laporan", "hasil_s2")  # bukan laporan/data/: pola data/ di .gitignore root

SUMMARIES = ["bench_gpu_summary.csv", "run_20261127_1634_ksweep_summary.csv",
             "run_20261127_1634_weak_summary.csv", "run_20261127_1634_cudavar_summary.csv"]
MESIN_B = "Mesin B"
TD = "--"  # sel tidak diukur; dijelaskan di setiap keterangan tabel

MACROS = {}


# ------------------------------------------------------------------ format
def num(v, d=1):
    """Angka desimal gaya Indonesia (koma), tanpa pemisah ribuan."""
    if v is None:
        return TD
    s = ("%." + str(d) + "f") % v
    if s.startswith("-"):
        s = "\\ensuremath{-}" + s[1:]
    return s.replace(".", ",")


def x(v, d=1):
    # \ensuremath: makro di angka.tex dipakai di teks biasa maupun di dalam mode matematika
    return TD if v is None else num(v, d) + "\\ensuremath{\\times}"


def pct(v, d=0):
    return TD if v is None else num(v, d) + "\\%"


def ribu(v):
    """Bilangan bulat dengan pemisah ribuan titik (gaya Indonesia)."""
    return "{:,}".format(int(v)).replace(",", ".")


def ha(v):
    """Luas hektare: pemisah ribuan titik, dua desimal koma."""
    a, b = ("%.2f" % v).split(".")
    return ribu(int(a)) + "," + b


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in MACROS, "makro ganda: " + name
    MACROS[name] = value


def img(r):
    return "%d$\\times$%d" % (r["width"], r["height"])


def tex_table(fname, caption, label, colspec, head, body, size="\\small", note=None, tabx=True):
    """Tabel gaya template: tabularx, garis abu-abu, kepala \\hdr, keterangan di atas, [H]."""
    lines = ["% DIBUAT OLEH laporan/skrip/tabel_ke_tex.py -- jangan disunting tangan.",
             "\\begin{table}[H]", "\\centering", "\\caption{%s}" % caption, "\\label{%s}" % label,
             size]
    env = "tabularx" if tabx else "tabular"
    lines.append("\\begin{%s}%s{%s}" % (env, "{\\textwidth}" if tabx else "", colspec))
    lines.append("\\hline")
    lines.append(" & ".join("\\hdr{%s}" % h for h in head) + " \\\\ \\hline")
    for row in body:
        if row == "MIDRULE":
            continue
        lines.append(" & ".join(str(c) for c in row) + " \\\\ \\hline")
    lines.append("\\end{%s}" % env)
    if note:
        lines.append("\\par\\smallskip\\footnotesize\\raggedright " + note)
    lines.append("\\end{table}")
    with open(os.path.join(OUT, fname), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ muat data Mesin B
rows = az.load([os.path.join(RES, s) for s in SUMMARIES])
assert not [r for r in rows if r.get("identik") == "BEDA"], "ada baris BEDA"
I = az.idx(rows)


def get(cfg, w, h, k, group="strong"):
    r = I.get((group, cfg, "sintetis_lanskap_%dx%d" % (w, h), k))
    assert r is not None, (cfg, w, h, k)
    return r


def best(prefix, r0, key="total_ms"):
    return az.best(rows, "strong", prefix, r0["image"], r0["k"], key)


SIZES_K = az.sizes_k(rows)  # (image, k) urut K lalu piksel

# Cek silang ke docs/BENCH_scaling_mesin_wisam.md (angka di dokumen tim, dibulatkan 0,1 ms)
assert round(get("L0", 4000, 3000, 8)["total_ms"], 1) == 2750.3  # run 16:34 menang atas bench_gpu (2603,2)
assert round(get("L1+L2 cuda x16", 7680, 4320, 8)["total_ms"], 1) == 684.1
_v = [r for r in rows if r["group"] == "cuda_variants" and r["config"] == "bench_cuda simple"
      and r["width"] == 7680 and r["k"] == 8]
assert len(_v) == 1 and round(_v[0]["kmeans_ms"], 1) == 1277.8

# ------------------------------------------------------------------ V.2 profil L0 Mesin B
body = []
porsi = []
for (im, k) in SIZES_K:
    r = I[("strong", "L0", im, k)]
    p = 100 * r["kmeans_ms"] / r["total_ms"]
    porsi.append(p)
    body.append([img(r), k, num(r["read_ms"]), num(r["init_ms"]), num(r["kmeans_ms"]), num(r["color_ms"]),
                 num(r["write_ms"]), num(r["total_ms"]), pct(p)])
tex_table("l0_mesinb.tex",
          "Waktu per tahap mode CPU saja (L0) dan porsi K-Means, citra sintetis, Mesin B (ms)",
          "tab:l0b", "L{2.3cm}r" + "Y" * 0 + ">{\\raggedleft\\arraybackslash}X" * 7,
          ["Citra", "$K$", "Baca", "Init", "K-Means", "Warna", "Tulis", "Total", "Porsi K-Means"], body,
          note="Median 5 ulangan setelah 1 pemanasan, 20 iterasi, \\emph{seed} 42. Sumber: "
               "\\kode{bench\\_gpu\\_summary.csv} dan \\kode{run\\_20261127\\_1634\\_ksweep\\_summary.csv} "
               "(4000$\\times$3000 dari \\emph{run} 16:34).")
macro("PorsiKMLoMin", pct(min(porsi)))
macro("PorsiKMLoMax", pct(max(porsi)))

# ------------------------------------------------------------------ Mesin A (pembanding berlabel)
md = open(os.path.join(ROOT, "docs", "PROFIL_L0_tabel.md"), encoding="utf-8").read()
body = []
porsiA = []
for line in md.splitlines():
    if not line.startswith("| sintetis_"):
        continue
    c = [s.strip() for s in line.strip("|").split("|")]
    w, h = (int(v) for v in c[1].split("x"))
    k = int(c[2])
    val = [float(s.split(" ")[0]) for s in c[3:8]]
    tot = float(c[8])
    porsiA.append(100 * val[2] / tot)
    body.append(["%d$\\times$%d" % (w, h), k] + [num(v) for v in val] + [num(tot), pct(porsiA[-1])])
assert len(body) == 6
tex_table("l0_mesina.tex",
          "Pembanding: waktu per tahap L0 di Mesin A (i7-1065G7), citra sintetis (ms)",
          "tab:l0a", "L{2.3cm}r" + ">{\\raggedleft\\arraybackslash}X" * 7,
          ["Citra", "$K$", "Baca", "Init", "K-Means", "Warna", "Tulis", "Total", "Porsi K-Means"], body,
          note="Median 5 ulangan setelah 1 pemanasan. Sumber: \\kode{docs/PROFIL\\_L0\\_tabel.md} "
               "(\\kode{bench/profile\\_l0.py}). Mesin berbeda dengan tabel lain di bab ini; "
               "angkanya tidak dibandingkan langsung dengan Mesin B.")
macro("PorsiKMAMin", pct(min(porsiA), 1))
macro("PorsiKMAMax", pct(max(porsiA), 1))


# ------------------------------------------------------------------ V.3 optimasi bertahap
def steps(im, k):
    l0 = I[("strong", "L0", im, k)]
    out = [("L0", l0), ("L2", best("L2 omp", l0)), ("L1", I[("strong", "L1 cuda", im, k)]),
           ("L1+L2", best("L1+L2 cuda", l0))]
    return l0, out


def short_cfg(r):
    c = r["config"]
    m = re.match(r"L2 omp x(\d+)", c)
    if m:
        return "L2 (%s thr)" % m.group(1)
    m = re.match(r"L1\+L2 cuda x(\d+)", c)
    if m:
        return "L1+L2 (%s thr)" % m.group(1)
    return {"L0": "L0", "L1 cuda": "L1"}[c]


def wall(r):
    return r.get("wall_ms")


def step_rows(pairs):
    body = []
    for im, k in pairs:
        l0, out = steps(im, k)
        for _, r in out:
            body.append([img(r), k, short_cfg(r), num(r["kmeans_ms"]), x(az.ratio(l0["kmeans_ms"], r["kmeans_ms"])),
                         num(r["total_ms"]), x(az.ratio(l0["total_ms"], r["total_ms"]), 2),
                         num(wall(r)), x(az.ratio(wall(l0), wall(r)), 2)])
    return body


HEAD3 = ["Citra", "$K$", "Konfigurasi", "K-Means", "$S$ K-Means", "Total", "$S$ Total", "Proses", "$S$ Proses"]
COL3 = "L{2.1cm}rL{2.3cm}" + ">{\\raggedleft\\arraybackslash}X" * 6
k8 = [(im, k) for (im, k) in SIZES_K if k == 8]
tex_table("bertahap_k8.tex",
          "Optimasi bertahap L0 $\\rightarrow$ L2 $\\rightarrow$ L1 $\\rightarrow$ L1+L2, $K=8$, citra sintetis, "
          "Mesin B (ms; $S$ = \\emph{speedup} terhadap L0)",
          "tab:bertahapk8", COL3, HEAD3, step_rows(k8), size="\\footnotesize",
          note="L2 dan L1+L2: konfigurasi terbaik menurut Total; ``thr'' = jumlah \\emph{thread} OpenMP. "
               "Total = \\kode{total\\_ms}, tanpa pemanasan konteks CUDA. Proses = waktu dinding satu proses "
               "\\kode{landcover} (\\kode{wall\\_ms}), termasuk pemanasan. ``--'' = tidak diukur: sesi "
               "\\kode{bench\\_gpu} (1000$^2$, 2000$^2$, 7680$\\times$4320) belum merekam \\kode{wall\\_ms}; "
               "baris 4000$\\times$3000 dari \\emph{run} 16:34.")
k4000 = [(im, k) for (im, k) in SIZES_K if im.endswith("4000x3000") and k != 8]
tex_table("bertahap_k.tex",
          "Optimasi bertahap pada 4000$\\times$3000 untuk $K=2$, 4, dan 16, citra sintetis, Mesin B "
          "(ms; $S$ = \\emph{speedup} terhadap L0)",
          "tab:bertahapk", COL3, HEAD3, step_rows(k4000), size="\\footnotesize",
          note="Keterangan kolom sama dengan Tabel~\\ref{tab:bertahapk8}. Sumber: "
               "\\kode{run\\_20261127\\_1634\\_ksweep\\_summary.csv}.")

# makro optimasi bertahap
l0_78, o78 = steps("sintetis_lanskap_7680x4320", 8)
d78 = dict(o78)
macro("LoTotalDelapan", num(l0_78["total_ms"]))
macro("LoKMDelapan", num(l0_78["kmeans_ms"]))
macro("LdSTotalDelapan", x(az.ratio(l0_78["total_ms"], d78["L1"]["total_ms"]), 2))
macro("LdLdSTotalDelapan", x(az.ratio(l0_78["total_ms"], d78["L1+L2"]["total_ms"]), 2))
macro("LdLdTotalDelapan", num(d78["L1+L2"]["total_ms"]))
macro("LdLdCfgDelapan", short_cfg(d78["L1+L2"]))
macro("LdSKMDelapan", x(az.ratio(l0_78["kmeans_ms"], d78["L1"]["kmeans_ms"])))
macro("LdKMDelapan", num(d78["L1"]["kmeans_ms"]))
macro("LdbSTotalDelapan", x(az.ratio(l0_78["total_ms"], d78["L2"]["total_ms"]), 2))
macro("LdbCfgDelapan", short_cfg(d78["L2"]))


def luar(r):
    return r["total_ms"] - r["kmeans_ms"]


macro("LuarLoDelapan", num(luar(l0_78)))
macro("LuarLdDelapan", num(luar(d78["L1"])))
macro("LuarLdLdDelapan", num(luar(d78["L1+L2"])))
r = d78["L1+L2"]
macro("BacaLdLdDelapan", num(r["read_ms"]))
macro("PorsiBacaLdLdDelapan", pct(100 * r["read_ms"] / r["total_ms"]))
macro("PorsiKMLdLdDelapan", pct(100 * r["kmeans_ms"] / r["total_ms"]))
macro("TulisLoDelapan", num(l0_78["write_ms"]))
macro("TulisLdLdDelapan", num(r["write_ms"]))
macro("BacaLoDelapan", num(l0_78["read_ms"]))

# Amdahl (Persamaan II.4) pada L0 7680x4320 K=8
p = l0_78["kmeans_ms"] / l0_78["total_ms"]
macro("AmdP", num(p, 3))
macro("AmdBatas", x(1 / (1 - p)))
macro("AmdBatasLdLd", x(r["total_ms"] / luar(r), 2))  # batas tambahan sesudah L1+L2 (BENCH_scaling 7)
assert abs(1 / (1 - p) - 7.6) < 0.05

# 4000x3000 K=8 (ada Proses)
l0_43, o43 = steps("sintetis_lanskap_4000x3000", 8)
d43 = dict(o43)
macro("LdSKMEmpat", x(az.ratio(l0_43["kmeans_ms"], d43["L1"]["kmeans_ms"])))
# klaim teks: menurut Total, L1+L2 tercepat pada 4000x3000 untuk semua K
for k in (2, 4, 8, 16):
    l0k, ok_ = steps("sintetis_lanskap_4000x3000", k)
    assert min(ok_, key=lambda t: t[1]["total_ms"])[0] == "L1+L2"
macro("LdLdSTotalEmpat", x(az.ratio(l0_43["total_ms"], d43["L1+L2"]["total_ms"]), 2))
macro("LdLdSProsesEmpat", x(az.ratio(wall(l0_43), wall(d43["L1+L2"])), 2))
macro("LdLdCfgEmpat", short_cfg(d43["L1+L2"]))
# konfigurasi Optimalisasi tercepat menurut Proses (semua omp dan cuda) pada 4000x3000, per K
optp = {}
for k in (2, 4, 8, 16):
    l0 = get("L0", 4000, 3000, k)
    cands = [q for q in rows if q["group"] == "strong" and q["image"] == l0["image"] and q["k"] == k
             and q["config"] != "L0" and q.get("wall_ms")]
    bp = min(cands, key=wall)
    optp[k] = (bp, wall(l0) / wall(bp))
names = {2: "Dua", 4: "Empat", 8: "Delapan", 16: "Enambelas"}
for k, (bp, s) in optp.items():
    macro("SProsesTerbaikK" + names[k], x(s, 2))
    macro("CfgProsesTerbaikK" + names[k], short_cfg(bp))
# klaim teks: hanya pada K=2 OpenMP tercepat menurut Proses
assert [k for k, (bp, _) in optp.items() if bp["config"].startswith("L2 omp")] == [2]
macro("SProsesMin", x(min(s for _, s in optp.values()), 2))
macro("SProsesMax", x(max(s for _, s in optp.values()), 2))
warm = [q["warmup_ms"] for q in rows if q.get("warmup_ms")]  # sesi 16:34; digabung dengan sesi ambang di bawah

# H2: K-Means saja, K=8, L1 vs L0 dan omp x8 vs L0 (semua ukuran K=8)
s_cuda = [az.ratio(I[("strong", "L0", im, 8)]["kmeans_ms"], I[("strong", "L1 cuda", im, 8)]["kmeans_ms"]) for im, _ in k8]
s_omp8 = [az.ratio(I[("strong", "L0", im, 8)]["kmeans_ms"], I[("strong", "L2 omp x8", im, 8)]["kmeans_ms"]) for im, _ in k8]
macro("HDuaCudaMin", x(min(s_cuda)))
macro("HDuaCudaMax", x(max(s_cuda)))
macro("HDuaOmpMin", x(min(s_omp8), 2))
macro("HDuaOmpMax", x(max(s_omp8), 2))
macro("HDuaOmpEmpat", x(az.ratio(l0_43["kmeans_ms"], get("L2 omp x8", 4000, 3000, 8)["kmeans_ms"]), 2))
for k in (2, 4, 16):
    macro("HDuaOmpK" + names[k], x(az.ratio(get("L0", 4000, 3000, k)["kmeans_ms"],
                                           get("L2 omp x8", 4000, 3000, k)["kmeans_ms"]), 2))
macro("HDuaOmpSatuEmpat", x(az.ratio(get("L2 omp x1", 4000, 3000, 8)["kmeans_ms"],
                                     get("L2 omp x8", 4000, 3000, 8)["kmeans_ms"]), 2))

# ------------------------------------------------------------------ V.4 strong scaling
body = []
for im, k in SIZES_K:
    b1 = I.get(("strong", "L2 omp x1", im, k))
    if not b1:
        continue
    line = [img(b1), k]
    sk = {}
    for t in (2, 4, 8, 16):
        q = I[("strong", "L2 omp x%d" % t, im, k)]
        sk[t] = (az.ratio(b1["kmeans_ms"], q["kmeans_ms"]), az.ratio(b1["total_ms"], q["total_ms"]))
    line += [x(sk[t][0], 2) for t in (2, 4, 8, 16)]
    line += [num(sk[8][0] / 8, 2), num(sk[16][0] / 16, 2)]
    line += [x(sk[8][1], 2), x(sk[16][1], 2)]
    body.append(line)
tex_table("strong.tex",
          "\\emph{Strong scaling} OpenMP: \\emph{speedup} $S$ terhadap OpenMP 1 \\emph{thread} dan efisiensi $E=S/T$, "
          "citra sintetis, Mesin B",
          "tab:strong", "L{2.1cm}r" + ">{\\raggedleft\\arraybackslash}X" * 8,
          ["Citra", "$K$", "K-M $T{=}2$", "K-M $T{=}4$", "K-M $T{=}8$", "K-M $T{=}16$", "$E$ K-M $T{=}8$",
           "$E$ K-M $T{=}16$", "Total $T{=}8$", "Total $T{=}16$"], body, size="\\footnotesize",
          note="K-M = tahap K-Means saja; Total = \\kode{total\\_ms} (termasuk baca PNG serial). "
               "Basis = \\kode{L2 omp x1}, bukan L0. Sumber: \\kode{docs/BENCH\\_scaling\\_mesin\\_wisam.md} bagian 3 "
               "(dihitung ulang dari CSV).")
# klaim teks: speedup K-Means naik monoton sampai 8 thread di semua ukuran
for im, k in SIZES_K:
    ks = [I[("strong", "L2 omp x%d" % t, im, k)]["kmeans_ms"] for t in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(ks, ks[1:])), (im, k)
better16 = []
for im, k in SIZES_K:
    a8 = I.get(("strong", "L2 omp x8", im, k))
    a16 = I.get(("strong", "L2 omp x16", im, k))
    if a8 and a16:
        better16.append((im, k, a16["kmeans_ms"] < a8["kmeans_ms"]))
macro("JmlStrong", str(len(better16)))
macro("JmlEnambelasMenang", str(sum(b for *_, b in better16)))
macro("DaftarEnambelasMenang", ", ".join("%s ($K=%d$)" % (img(I[("strong", "L0", im, k)]), k)
                                          for im, k, b in better16 if b))
b1 = get("L2 omp x1", 7680, 4320, 8)
macro("StrongDelapanTd", x(az.ratio(b1["kmeans_ms"], get("L2 omp x8", 7680, 4320, 8)["kmeans_ms"]), 2))
macro("StrongDelapanTe", x(az.ratio(b1["kmeans_ms"], get("L2 omp x16", 7680, 4320, 8)["kmeans_ms"]), 2))
macro("OmpSatuLoKMDelapan", x(az.ratio(l0_78["kmeans_ms"], b1["kmeans_ms"]), 2))

# ------------------------------------------------------------------ V.5 weak scaling
weak = [q for q in rows if q["group"] == "weak"]
body = []
ew = {}
for t in (1, 2, 4, 8, 16):
    line = None
    for k in (4, 8):
        q = next(r_ for r_ in weak if r_["k"] == k and r_["threads"] == t)
        b = next(r_ for r_ in weak if r_["k"] == k and r_["threads"] == 1)
        if line is None:
            line = [t, img(q)]
        e_km, e_tot = b["kmeans_ms"] / q["kmeans_ms"], b["total_ms"] / q["total_ms"]
        ew[(t, k)] = (e_km, e_tot)
        line += [num(q["kmeans_ms"]), num(e_km, 2), num(q["total_ms"]), num(e_tot, 2)]
    body.append(line)
tex_table("weak.tex",
          "\\emph{Weak scaling} OpenMP (beban per \\emph{thread} tetap 2 juta piksel), citra sintetis, Mesin B (ms)",
          "tab:weak", "rL{2.4cm}" + ">{\\raggedleft\\arraybackslash}X" * 8,
          ["$T$", "Citra", "K-M $K{=}4$", "$E_w$", "Total $K{=}4$", "$E_w$",
           "K-M $K{=}8$", "$E_w$", "Total $K{=}8$", "$E_w$"], body, size="\\footnotesize",
          note="$E_w = t(T{=}1)/t(T)$, ideal 1. Sumber: \\kode{run\\_20261127\\_1634\\_weak\\_summary.csv}.")
macro("WeakKMEnambelas", num(ew[(16, 8)][0], 2))
macro("WeakTotEnambelas", num(ew[(16, 8)][1], 2))
macro("WeakKMDelapan", num(ew[(8, 8)][0], 2))
macro("WeakKMDua", num(ew[(2, 8)][0], 2))

# ------------------------------------------------------------------ V.6 CUDA: rincian, pengaruh K, throughput, varian
body = []
for im, k in SIZES_K:
    q = I[("strong", "L1 cuda", im, k)]
    body.append([img(q), k, num(q["h2d_ms"]), num(q["kernel_ms"]), num(q["d2h_ms"]), num(q["kmeans_ms"]),
                 num(q.get("warmup_ms") or None, 0)])
tex_table("cuda_rinci.tex",
          "Rincian tahap K-Means CUDA (L1) per citra dan $K$, citra sintetis, Mesin B (ms)",
          "tab:cudarinci", "L{2.3cm}r" + ">{\\raggedleft\\arraybackslash}X" * 5,
          ["Citra", "$K$", "h2d", "\\emph{Kernel}", "d2h", "K-Means", "Pemanasan"], body,
          note="h2d/\\emph{kernel}/d2h diukur dengan CUDA \\emph{events}. Pemanasan = inisialisasi konteks CUDA "
               "(\\kode{warmup\\_ms}), di luar \\kode{total\\_ms}; ``--'' = tidak direkam pada sesi \\kode{bench\\_gpu}.")
# klaim teks: pemanasan > tahap K-Means pada semua baris L1 yang mencatatnya; d2h > h2d
for im, k in SIZES_K:
    q = I[("strong", "L1 cuda", im, k)]
    if q.get("warmup_ms"):
        assert q["warmup_ms"] > q["kmeans_ms"]
    assert q["d2h_ms"] > q["h2d_ms"]
rasio = [I[("strong", "L1 cuda", im, k)]["d2h_ms"] / I[("strong", "L1 cuda", im, k)]["h2d_ms"] for im, k in SIZES_K]
macro("RasioDdhHdMin", x(min(rasio)))
macro("RasioDdhHdMax", x(max(rasio)))
kern = [get("L1 cuda", 4000, 3000, k)["kernel_ms"] for k in (2, 4, 8, 16)]
macro("KernelKMin", num(min(kern)))
macro("KernelKMax", num(max(kern)))
warm4 = [get("L1 cuda", 4000, 3000, k)["warmup_ms"] for k in (2, 4, 8, 16)]
assert all(warm4) and not any(I[("strong", "L1 cuda", im, k)].get("warmup_ms") for im, k in SIZES_K
                              if not im.endswith("4000x3000"))  # klaim teks: hanya 4000x3000 mencatat pemanasan
q = get("L1 cuda", 7680, 4320, 8)
macro("HdDelapan", num(q["h2d_ms"]))
macro("KernelDelapan", num(q["kernel_ms"]))
macro("DdhDelapan", num(q["d2h_ms"]))

# Pengaruh K pada 4000x3000
body = []
for cfg in ("L0", "L2 omp x8", "L2 omp x16", "L1 cuda", "L1+L2 cuda x8"):
    base = get(cfg, 4000, 3000, 2)["kmeans_ms"]
    line = [short_cfg(get(cfg, 4000, 3000, 2))]
    for k in (2, 4, 8, 16):
        v = get(cfg, 4000, 3000, k)["kmeans_ms"]
        line.append("%s (%s)" % (num(v), x(v / base, 2)))
    body.append(line)
tex_table("pengaruh_k.tex",
          "Waktu K-Means (ms) terhadap $K$ pada 4000$\\times$3000, citra sintetis, Mesin B "
          "(dalam kurung: rasio terhadap $K=2$)",
          "tab:pengaruhk", "L{2.6cm}" + ">{\\raggedleft\\arraybackslash}X" * 4,
          ["Konfigurasi", "$K=2$", "$K=4$", "$K=8$", "$K=16$"], body,
          note="Sumber: \\kode{run\\_20261127\\_1634\\_ksweep\\_summary.csv}.")
macro("RasioKLo", x(get("L0", 4000, 3000, 16)["kmeans_ms"] / get("L0", 4000, 3000, 2)["kmeans_ms"], 2))
macro("RasioKCuda", x(get("L1 cuda", 4000, 3000, 16)["kmeans_ms"] / get("L1 cuda", 4000, 3000, 2)["kmeans_ms"], 2))
macro("LdSKMEnambelas", x(az.ratio(get("L0", 4000, 3000, 16)["kmeans_ms"], get("L1 cuda", 4000, 3000, 16)["kmeans_ms"])))


def nspx(q):
    return q["kmeans_ms"] * 1e6 / (q["px"] * q["iters"])


body = []
tp = {}
for im, k in k8:
    line = [img(I[("strong", "L0", im, k)])]
    line.append(num(I[("strong", "L0", im, k)]["px"] / 1e6, 2))
    for cfg in ("L0", "L2 omp x8", "L1 cuda"):
        v = nspx(I[("strong", cfg, im, k)])
        tp[(im, cfg)] = v
        line.append(num(v, 2))
    body.append(line)
tex_table("throughput.tex",
          "\\emph{Throughput} K-Means dalam ns per piksel per iterasi, $K=8$, citra sintetis, Mesin B",
          "tab:throughput", "L{2.6cm}" + ">{\\raggedleft\\arraybackslash}X" * 4,
          ["Citra", "Piksel (juta)", "L0", "L2 (8 thr)", "L1"], body,
          note="ns/piksel/iterasi = \\kode{kmeans\\_ms}$\\times10^6$/(piksel$\\times$iterasi). "
               "Sumber: \\kode{docs/BENCH\\_scaling\\_mesin\\_wisam.md} bagian 5 (dihitung ulang dari CSV).")
cu = [v for (im, c), v in tp.items() if c == "L1 cuda"]
macro("TpCudaMin", num(min(cu), 2))
macro("TpCudaMax", num(max(cu), 2))

# Varian CUDA T3/T4/T5
var = [q for q in rows if q["group"] == "cuda_variants"]
keys = sorted({(q["px"], q["k"]) for q in var})
body = []
t4t3, t5t4 = [], []
for px, k in keys:
    def v(c):
        return next(q for q in var if q["config"] == "bench_cuda " + c and q["px"] == px and q["k"] == k)
    s, sh, st = v("simple"), v("shared"), v("streams")
    t4t3.append(s["kmeans_ms"] / sh["kmeans_ms"])
    t5t4.append(st["kmeans_ms"] / sh["kmeans_ms"])
    body.append([img(s), k, num(s["kmeans_ms"]), num(sh["kmeans_ms"]), num(st["kmeans_ms"]),
                 x(t4t3[-1]), x(s["kmeans_ms"] / st["kmeans_ms"])])
tex_table("varian_cuda.tex",
          "Varian CUDA pada tahap K-Means saja (\\kode{tests/bench\\_cuda}, citra acak), Mesin B (ms)",
          "tab:variancuda", "L{2.3cm}r" + ">{\\raggedleft\\arraybackslash}X" * 5,
          ["Ukuran", "$K$", "T3 \\emph{simple}", "T4 \\emph{shared}", "T5 \\emph{streams}", "T4 vs T3",
           "T5 vs T3"], body,
          note="Diukur di luar \\kode{landcover} (tanpa baca/tulis PNG). Sumber: "
               "\\kode{run\\_20261127\\_1634\\_cudavar\\_summary.csv}.")
assert min(t5t4) > 1.0  # klaim teks: T5 lebih lambat daripada T4 di semua ukuran
macro("TempatTigaMin", x(min(t4t3)))
macro("TempatTigaMax", x(max(t4t3)))
macro("LimaEmpatMin", x(min(t5t4), 2))
macro("LimaEmpatMax", x(max(t5t4), 2))

# ------------------------------------------------------------------ ambang mode otomatis
amb_csv = os.path.join(RES, "ambang_summary.csv")
cases = fat.load(amb_csv)
assert len(cases) == 21
body = []
T = 70.0
for c in cases:
    win_t = "cuda" if c["ct"] < c["ot"] else "omp"
    win_p = "cuda" if c["cw"] < c["ow"] else "omp"
    rule = "cuda" if c["n"] * c["k"] >= T * 1e6 else "omp"
    body.append([num(c["n"] / 1e6, 2), c["k"], num(c["n"] * c["k"] / 1e6, 1), num(c["ot"]), num(c["ct"]),
                 win_t, num(c["ow"]), num(c["cw"]), win_p, rule])
tex_table("ambang.tex",
          "Pemenang OpenMP vs CUDA menurut Total dan menurut Proses pada 21 konfigurasi, citra sintetis, Mesin B (ms)",
          "tab:ambang", "rr" + ">{\\raggedleft\\arraybackslash}X" * 8,
          ["$N$ (juta)", "$K$", "$N{\\times}K$ (juta)", "omp Total", "cuda Total", "Unggul", "omp Proses",
           "cuda Proses", "Unggul", "Aturan"], body, size="\\footnotesize",
          note="omp = \\kode{L2 omp x8}; cuda = yang tercepat dari L1 dan L1+L2 (8 \\emph{thread}) untuk "
               "kriteria itu. Proses = waktu dinding, termasuk pemanasan CUDA. Aturan = pilihan "
               "$N\\times K\\geq70$ juta. Sumber: \\kode{bench/results/ambang\\_summary.csv} "
               "(\\kode{bench/fit\\_auto\\_threshold.py}).")
rules = {"Lama": lambda n, k: n >= 100000, "Baru": lambda n, k: n * k >= T * 1e6, "Omp": lambda n, k: False}
res = {name: (fat.score(cases, f), fat.score(cases, f, False)) for name, f in rules.items()}
# cek silang dengan docs/AMBANG_AUTO.md
assert res["Lama"][0][0] == 5 and round(res["Lama"][0][1]) == 2585
assert res["Baru"][0] == (21, 0) and res["Baru"][1][0] == 5
assert res["Omp"][0][0] == 16 and round(res["Omp"][0][1]) == 1427
for name, ((okp, rgp), (okt, _)) in res.items():
    macro("Amb%sBenarProses" % name, str(okp))
    macro("Amb%sRugi" % name, ribu(round(rgp)))
    macro("Amb%sBenarTotal" % name, str(okt))
macro("AmbJumlah", str(len(cases)))
macro("AmbCudaTotal", str(sum(1 for c in cases if c["ct"] < c["ot"])))
macro("AmbCudaProses", str(sum(1 for c in cases if c["cw"] < c["ow"])))
macro("AmbNMax", num(max(c["n"] for c in cases) / 1e6, 0))
amb_rows = list(__import__("csv").DictReader(open(amb_csv, encoding="utf-8")))
wa = [float(q["warmup_ms"]) for q in amb_rows if q["mode"] == "cuda"]
macro("AmbWarmMin", num(min(wa), 0))
macro("AmbWarmMax", num(max(wa), 0))
# pemanasan pada semua sesi Mesin B yang merekamnya (run 16:34 + sesi ambang)
macro("WarmMin", num(min(warm + wa), 0))
macro("WarmMax", num(max(warm + wa), 0))

# ------------------------------------------------------------------ verifikasi (kolom identik benchmark)
allsum = rows + [dict(q) for q in amb_rows]
chk = [q for q in allsum if q.get("identik") in ("ya", "tidak", "BEDA")]
macro("IdentikYa", str(sum(1 for q in chk if q["identik"] == "ya")))
macro("IdentikJumlah", str(len(chk)))
assert all(q["identik"] == "ya" for q in chk)
cmp_lines = [ln.split() for ln in open(os.path.join(DATA_S2, "cmp.txt"), encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
assert all(ln[3] == "SAMA" for ln in cmp_lines) and len(cmp_lines) == 6
macro("CmpSama", str(len(cmp_lines)))

# ------------------------------------------------------------------ Sentinel-2 MDM 2048 px, K=8 (Mesin C)
counts = json.load(open(os.path.join(DATA_S2, "counts.json"), encoding="utf-8"))
cents = json.load(open(os.path.join(DATA_S2, "centroids.json"), encoding="utf-8"))
meta = json.load(open(os.path.join(DATA_S2, "meta_cpu.json"), encoding="utf-8"))
kelas = json.load(open(os.path.join(DATA_S2, "kelas.json"), encoding="utf-8"))
npx = meta["width"] * meta["height"]
nod, nodc = meta["nodata_pixels"], meta["nodata_cluster"]
assert sum(counts) == npx and counts[nodc] >= nod
# cek silang dengan PROGRESS Sesi U (indeks kelompok stabil antarbiner)
assert nod == 270420 and counts[1] == 114105
valid = npx - nod
hapx = meta["pixel_size_m"] ** 2 / 1e4
body = []
seen = set()
for nama, idxs in kelas["kelas"]:
    n_ = sum(counts[i] - (nod if i == nodc else 0) for i in idxs)
    seen.update(idxs)
    kel = ", ".join("%d (%s)" % (i, ",".join(str(c) for c in cents[i])) for i in idxs)
    body.append([nama, kel, ribu(n_), pct(100 * n_ / valid, 2), ha(n_ * hapx)])
assert seen | {nodc} == set(range(meta["k"]))
body.append(["\\textbf{Total valid}", "", ribu(valid), pct(100.0, 2), ha(valid * hapx)])
body.append(["\\emph{Nodata} (dikecualikan)", "%d (0,0,0)" % nodc, ribu(nod), pct(100 * nod / npx, 2),
             ha(nod * hapx)])
tex_table("luas_s2.tex",
          "Persentase dan luas per kelas, Sentinel-2 T49MDM 2048$\\times$2048 px, $K=8$",
          "tab:luass2", "L{2.9cm}Y>{\\raggedleft\\arraybackslash}p{2.1cm}>{\\raggedleft\\arraybackslash}p{1.6cm}"
                        ">{\\raggedleft\\arraybackslash}p{1.9cm}",
          ["Kelas", "Kelompok (\\emph{centroid} R,G,B)", "Piksel", "Persen", "Luas (ha)"], body,
          note="Persen kelas dihitung dari piksel valid; persen \\emph{nodata} dari seluruh citra. "
               "1 piksel = %s m $\\times$ %s m. Nama kelas diberikan manual dari warna \\emph{centroid} dan "
               "kenampakan visual (bukan klasifikasi terverifikasi). Sumber: \\kode{laporan/hasil\\_s2/} "
               "(\\kode{counts.json}, \\kode{meta\\_cpu.json}, \\kode{kelas.json}); Mesin C." %
               (meta["pixel_size_m"], meta["pixel_size_m"]))
macro("SNodataPersen", pct(100 * nod / npx, 2))
macro("SNodataPiksel", ribu(nod))
macro("SValidHa", ha(valid * hapx))
macro("SPiksel", ribu(npx))
macro("SKelompokNodata", str(nodc))

# ------------------------------------------------------------------ tulis angka.tex
with open(os.path.join(OUT, "angka.tex"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("% DIBUAT OLEH laporan/skrip/tabel_ke_tex.py -- jangan disunting tangan.\n"
             "% Angka yang dikutip di teks Bab V (dan VI). Sumber: lihat docstring skrip.\n")
    for k_ in sorted(MACROS):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k_, MACROS[k_]))
print("tulis %d tabel dan %d makro ke %s" % (len([f for f in os.listdir(OUT) if f.endswith('.tex')]) - 1,
                                             len(MACROS), os.path.relpath(OUT, ROOT)))
