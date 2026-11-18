// test_cuda_vs_ref.cpp - kmeans_cuda harus IDENTIK (label dan centroid) dengan ref_kmeans.
//
// Pemakaian:
//   test_cuda_vs_ref            uji kecil-menengah (ukuran x K x jenis citra)
//   test_cuda_vs_ref --small    hanya citra kecil (untuk compute-sanitizer)
//   test_cuda_vs_ref --large    tambah 7680x4320 K=8 dan uji muat 10980x10980 K=8
//
// Di T1 berkas ini sengaja gagal di tahap link karena kmeans_cuda belum ada (dibuat di T3).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kmeans.h"
#include "ref_kmeans.h"
#include "test_util.h"

using namespace testutil;

static int g_fail = 0, g_total = 0;

static bool run_case(ImageKind kind, int w, int h, int k, int iters, bool verbose) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    auto img = make_image(kind, w, h, 1234);
    KMeansParams p{};
    p.width = w; p.height = h; p.k = k; p.max_iter = iters; p.seed = 42; p.num_threads = 1;

    std::vector<int32_t> lr(n), lc(n, -1);
    std::vector<uint8_t> cr(k * 3), cc(k * 3, 0);
    StageTimes tr{}, tc{};
    char tag[160];
    std::snprintf(tag, sizeof(tag), "%s %dx%d K=%d iter=%d", kind_name(kind), w, h, k, iters);

    ++g_total;
    if (ref_kmeans(img.data(), lr.data(), cr.data(), &p, &tr) != 0) {
        std::printf("[GAGAL] %s: ref_kmeans galat\n", tag); ++g_fail; return false;
    }
    if (kmeans_cuda(img.data(), lc.data(), cc.data(), &p, &tc) != 0) {
        std::printf("[GAGAL] %s: kmeans_cuda galat\n", tag); ++g_fail; return false;
    }
    const bool ok = same_result(lr, cr, lc, cc, tag);
    if (!ok) ++g_fail;
    if (!ok || verbose) {
        std::printf("[%s] %s  (ref %.1f ms, cuda total %.1f ms, kernel %.1f ms)\n",
                    ok ? "LULUS" : "GAGAL", tag, tr.total_ms, tc.total_ms, tc.kernel_ms);
    }
    return ok;
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

    // Satu iterasi: label harus sesuai centroid awal (kontrak butir 3.7).
    run_case(ImageKind::Random, 64, 64, 8, 1, false);

    if (large) {
        run_case(ImageKind::Random, 7680, 4320, 8, 20, true);
        run_case(ImageKind::Gradient, 10980, 10980, 8, 20, true);  // uji muat memori GPU 6 GB
    }

    std::printf("%s: %d dari %d kasus lulus\n", g_fail ? "GAGAL" : "SEMUA LULUS",
                g_total - g_fail, g_total);
    return g_fail ? 1 : 0;
}
