// bench_cuda.cpp - benchmark mode "cuda" pada citra sintetis, satu baris CSV sesuai kontrak butir 4.
//
//   bench_cuda --variant shared --width 7680 --height 4320 --k 8 --iters 20 --reps 5
//              --out bench/results/cuda_shared.csv          (satu baris perintah)
//
// --variant shared = kmeans_cuda (T4, versi utama), simple = kmeans_cuda_simple (T3).
// Kolom mode tetap "cuda" sesuai kontrak; varian dibedakan lewat nama berkas keluaran.
//
// 1 kali pemanasan, lalu --reps kali pengukuran; tiap kolom waktu dilaporkan sebagai median.
// Citra sintetis tidak dibaca dari disk, jadi read_ms, color_ms, write_ms = 0.
// total_ms = waktu dinding panggilan kmeans_cuda (termasuk alokasi dan transfer).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kmeans.h"
#include "kmeans_cuda_variants.h"
#include "test_util.h"

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int main(int argc, char** argv) {
    int w = 7680, h = 4320, k = 8, iters = 20, reps = 5;
    uint32_t seed = 42;
    std::string out = "bench/results/cuda_shared.csv";
    std::string variant = "shared";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const char* v = argv[i + 1];
        if (a == "--width") w = std::atoi(v);
        else if (a == "--height") h = std::atoi(v);
        else if (a == "--k") k = std::atoi(v);
        else if (a == "--iters") iters = std::atoi(v);
        else if (a == "--reps") reps = std::atoi(v);
        else if (a == "--seed") seed = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        else if (a == "--out") out = v;
        else if (a == "--variant") variant = v;
        else { std::fprintf(stderr, "argumen tidak dikenal: %s\n", a.c_str()); return 2; }
    }
    if (reps < 1) reps = 1;
    int (*fn)(const uint8_t*, int32_t*, uint8_t*, const KMeansParams*, StageTimes*) = nullptr;
    if (variant == "shared") fn = kmeans_cuda;
    else if (variant == "simple") fn = kmeans_cuda_simple;
    else if (variant == "streams") fn = kmeans_cuda_streams;
    else { std::fprintf(stderr, "varian tidak dikenal: %s (shared|simple|streams)\n", variant.c_str()); return 2; }
    const char* mode = (variant == "streams") ? "cuda_streams" : "cuda";

    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    auto img = testutil::make_image(testutil::ImageKind::Random, w, h, 1234);
    std::vector<int32_t> labels(n);
    std::vector<uint8_t> cen(static_cast<size_t>(k) * 3);
    KMeansParams p{};
    p.width = w; p.height = h; p.k = k; p.max_iter = iters; p.seed = seed; p.num_threads = 0;

    std::vector<double> init, km, tot, h2d, ker, d2h;
    for (int r = 0; r <= reps; ++r) {  // r == 0: pemanasan, tidak dicatat
        StageTimes t{};
        const auto t0 = std::chrono::steady_clock::now();
        if (fn(img.data(), labels.data(), cen.data(), &p, &t) != 0) {
            std::fprintf(stderr, "kmeans_cuda (%s) galat\n", variant.c_str());
            return 1;
        }
        const double wall =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "%s %d: wall=%.3f kmeans=%.3f h2d=%.3f kernel=%.3f d2h=%.3f\n",
                     r == 0 ? "pemanasan" : "ulang", r, wall, t.kmeans_ms, t.h2d_ms, t.kernel_ms,
                     t.d2h_ms);
        if (r == 0) continue;
        init.push_back(t.init_ms); km.push_back(t.kmeans_ms); tot.push_back(wall);
        h2d.push_back(t.h2d_ms); ker.push_back(t.kernel_ms); d2h.push_back(t.d2h_ms);
    }

    FILE* f = std::fopen(out.c_str(), "r");
    const bool need_header = (f == nullptr);
    if (f) std::fclose(f);
    f = std::fopen(out.c_str(), "a");
    if (!f) { std::fprintf(stderr, "tidak bisa menulis %s\n", out.c_str()); return 1; }
    if (need_header) {
        std::fprintf(f, "mode,image,width,height,k,iters,threads,read_ms,init_ms,kmeans_ms,"
                        "color_ms,write_ms,total_ms,h2d_ms,kernel_ms,d2h_ms\n");
    }
    char line[512];
    std::snprintf(line, sizeof(line),
                  "%s,synthetic_random,%d,%d,%d,%d,0,0,%.3f,%.3f,0,0,%.3f,%.3f,%.3f,%.3f\n", mode, w,
                  h, k, iters, median(init), median(km), median(tot), median(h2d), median(ker),
                  median(d2h));
    std::fputs(line, f);
    std::fclose(f);
    std::printf("%s", line);
    std::printf("(varian %s, median %d kali setelah 1 pemanasan, ditambahkan ke %s)\n",
                variant.c_str(), reps, out.c_str());
    return 0;
}
