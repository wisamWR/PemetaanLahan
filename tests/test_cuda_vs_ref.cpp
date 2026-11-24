// test_cuda_vs_ref.cpp - kmeans_cuda (T4) dan kmeans_cuda_simple (T3) harus IDENTIK
// (label dan centroid) dengan ref_kmeans. Tiap kasus dihitung sebagai dua uji (satu per varian).
//
// Pemakaian:
//   test_cuda_vs_ref            uji kecil-menengah (ukuran x K x jenis citra)
//   test_cuda_vs_ref --small    hanya citra kecil (untuk compute-sanitizer)
//   test_cuda_vs_ref --large    tambah 7680x4320 K=8 dan uji muat 10980x10980 K=8
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kmeans.h"
#include "kmeans_cuda_variants.h"
#include "ref_kmeans.h"
#include "test_util.h"

using namespace testutil;

typedef int (*KMeansFn)(const uint8_t*, int32_t*, uint8_t*, const KMeansParams*, StageTimes*);

static int g_fail = 0, g_total = 0;

static bool run_case(ImageKind kind, int w, int h, int k, int iters, bool verbose) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    auto img = make_image(kind, w, h, 1234);
    KMeansParams p{};
    p.width = w; p.height = h; p.k = k; p.max_iter = iters; p.seed = 42; p.num_threads = 1;

    std::vector<int32_t> lr(n);
    std::vector<uint8_t> cr(k * 3);
    StageTimes tr{};
    char base[160];
    std::snprintf(base, sizeof(base), "%s %dx%d K=%d iter=%d", kind_name(kind), w, h, k, iters);

    if (ref_kmeans(img.data(), lr.data(), cr.data(), &p, &tr) != 0) {
        std::printf("[GAGAL] %s: ref_kmeans galat\n", base); ++g_total; ++g_fail; return false;
    }

    // Kedua varian CUDA dibandingkan dengan referensi yang sama.
    struct Variant { const char* name; KMeansFn fn; };
    const Variant variants[] = {{"simple(T3)", kmeans_cuda_simple}, {"shared(T4)", kmeans_cuda}};
    bool all_ok = true;
    for (const auto& v : variants) {
        std::vector<int32_t> lc(n, -1);
        std::vector<uint8_t> cc(k * 3, 0);
        StageTimes tc{};
        const std::string tag = std::string(v.name) + " " + base;
        ++g_total;
        if (v.fn(img.data(), lc.data(), cc.data(), &p, &tc) != 0) {
            std::printf("[GAGAL] %s: galat\n", tag.c_str()); ++g_fail; all_ok = false; continue;
        }
        const bool ok = same_result(lr, cr, lc, cc, tag);
        if (!ok) { ++g_fail; all_ok = false; }
        if (!ok || verbose) {
            std::printf("[%s] %s  (ref %.1f ms, cuda total %.1f ms, kernel %.1f ms)\n",
                        ok ? "LULUS" : "GAGAL", tag.c_str(), tr.total_ms, tc.total_ms, tc.kernel_ms);
        }
    }

    // L3 streams (T5): beberapa ukuran potongan, termasuk potongan sisa yang tidak penuh.
    const size_t chunks[] = {0, n <= 70000 ? size_t(97) : n / 7 + 3, n / 2 + 1};
    for (size_t chunk : chunks) {
        std::vector<int32_t> lc(n, -1);
        std::vector<uint8_t> cc(k * 3, 0);
        StageTimes tc{};
        const std::string tag = "streams(T5,chunk=" + std::to_string(chunk) + ") " + base;
        ++g_total;
        if (kmeans_cuda_streams_chunked(img.data(), lc.data(), cc.data(), &p, &tc, chunk) != 0) {
            std::printf("[GAGAL] %s: galat\n", tag.c_str()); ++g_fail; all_ok = false; continue;
        }
        const bool ok = same_result(lr, cr, lc, cc, tag);
        if (!ok) { ++g_fail; all_ok = false; }
        if (!ok || (verbose && chunk == 0)) {
            std::printf("[%s] %s  (cuda_streams total %.1f ms)\n", ok ? "LULUS" : "GAGAL",
                        tag.c_str(), tc.total_ms);
        }
    }
    return all_ok;
}

int main(int argc, char** argv) {
    bool large = false, small_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--large") == 0) large = true;
        else if (std::strcmp(argv[i], "--small") == 0) small_only = true;
    }

    struct Size { int w, h; };
    // 257x3 dan 1023x7: jumlah piksel bukan kelipatan ukuran blok.
    std::vector<Size> sizes = {{1, 1}, {17, 13}, {64, 64}, {257, 3}, {1023, 7}};
    if (!small_only) sizes.push_back({1000, 1000});
    const int ks[] = {1, 2, 4, 8, 16};
    const ImageKind kinds[] = {ImageKind::Random, ImageKind::Solid, ImageKind::FewColors,
                               ImageKind::Gradient};

    for (const auto& s : sizes)
        for (int k : ks)
            for (ImageKind kind : kinds) run_case(kind, s.w, s.h, k, 20, false);

    // Satu dan dua iterasi: label harus sesuai centroid awal (kontrak butir 3.7); untuk streams,
    // iterasi pertama sekaligus terakhir (1) dan tanpa iterasi tengah (2).
    run_case(ImageKind::Random, 64, 64, 8, 1, false);
    run_case(ImageKind::Random, 64, 64, 8, 2, false);
    run_case(ImageKind::Random, 1000, 1000, 8, 1, false);

    if (large) {
        run_case(ImageKind::Random, 7680, 4320, 8, 20, true);
        run_case(ImageKind::Gradient, 10980, 10980, 8, 20, true);  // uji muat memori GPU 6 GB
    }

    std::printf("%s: %d dari %d kasus lulus\n", g_fail ? "GAGAL" : "SEMUA LULUS",
                g_total - g_fail, g_total);
    return g_fail ? 1 : 0;
}
