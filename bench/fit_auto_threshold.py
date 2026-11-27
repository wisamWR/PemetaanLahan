#!/usr/bin/env python3
"""Menurunkan dan menguji ambang --kmeans auto dari bench/results/ambang_summary.csv.

Per (ukuran, K) diambil konfigurasi omp terbaik dan cuda terbaik. Dua kriteria:
  Proses = wall_ms (satu proses CLI, termasuk pemanasan CUDA): yang dirasakan pengguna
  Total  = total_ms (tanpa pemanasan CUDA)
Aturan yang diuji: lama (N >= 100.000) dan baru (N*K >= T). Mencetak kecocokan, rugi waktu
(jumlah ms bila memilih mesin yang bukan tercepat), dan rentang T terbaik.

  python3 bench/fit_auto_threshold.py [--csv bench/results/ambang_summary.csv] [--t 70]
"""
import argparse, collections, csv


def load(path):
    cases = collections.defaultdict(lambda: {"omp": [], "cuda": []})
    for r in csv.DictReader(open(path, encoding="utf-8")):
        side = "cuda" if r["mode"] == "cuda" else "omp"
        cases[(int(r["width"]) * int(r["height"]), int(r["k"]))][side].append(
            (float(r["total_ms"]), float(r["wall_ms"])))
    out = []
    for (n, k), d in sorted(cases.items(), key=lambda x: (x[0][1], x[0][0])):
        if d["omp"] and d["cuda"]:
            out.append(dict(n=n, k=k, ot=min(t for t, _ in d["omp"]), ct=min(t for t, _ in d["cuda"]),
                            ow=min(w for _, w in d["omp"]), cw=min(w for _, w in d["cuda"])))
    return out


def score(data, rule, wall=True):
    ok = regret = 0
    for c in data:
        o, g = (c["ow"], c["cw"]) if wall else (c["ot"], c["ct"])
        got = g if rule(c["n"], c["k"]) else o
        ok += got == min(o, g)
        regret += got - min(o, g)
    return ok, regret


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="bench/results/ambang_summary.csv")
    ap.add_argument("--t", type=float, default=70.0, help="ambang N*K (juta) yang diuji")
    a = ap.parse_args()
    data = load(a.csv)
    print("konfigurasi terukur:", len(data))
    rules = [("lama: N >= 100.000 px", lambda n, k: n >= 100000),
             ("baru: N*K >= %g juta" % a.t, lambda n, k: n * k >= a.t * 1e6),
             ("selalu omp", lambda n, k: False), ("selalu cuda", lambda n, k: True)]
    print("%-26s %-34s %s" % ("aturan", "Proses (benar, rugi ms)", "Total (benar)"))
    for name, rule in rules:
        p, t = score(data, rule), score(data, rule, False)
        print("%-26s %2d/%d, rugi %7.0f ms          %2d/%d" % (name, p[0], len(data), p[1], t[0], len(data)))
    ts = [t for t in range(5, 300) if score(data, lambda n, k, t=t: n * k >= t * 1e6) == max(
        score(data, lambda n, k, u=u: n * k >= u * 1e6) for u in range(5, 300))]
    print("\nT terbaik (Proses): %d-%d juta piksel x cluster" % (ts[0], ts[-1]) if ts else "")
    print("\n n(juta)   K  N*K(juta)  omp(ms)  cuda(ms)  tercepat  aturan-baru  aturan-lama")
    for c in data:
        print("%7.2f  %2d  %9.1f  %7.1f  %8.1f  %-8s  %-11s  %s" % (
            c["n"] / 1e6, c["k"], c["n"] * c["k"] / 1e6, c["ow"], c["cw"], "cuda" if c["cw"] < c["ow"] else "omp",
            "cuda" if c["n"] * c["k"] >= a.t * 1e6 else "omp", "cuda" if c["n"] >= 1e5 else "omp"))


if __name__ == "__main__":
    main()
