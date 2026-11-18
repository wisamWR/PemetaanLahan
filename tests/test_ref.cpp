// test_ref.cpp - uji referensi CPU sementara (T1):
//   1. determinisme: dua kali jalan dengan parameter sama -> label dan centroid identik
//   2. jawaban yang diketahui: citra dua warna, K=2 -> centroid = dua warna itu
//   3. kasus tepi inisialisasi (k > n) dan penolakan argumen tidak sah
#include <cstdio>
#include <set>
#include <vector>

#include "ref_kmeans.h"
#include "test_util.h"

using namespace testutil;

static int g_fail = 0;

static void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "LULUS" : "GAGAL", what);
    if (!ok) ++g_fail;
}

static KMeansParams params(int w, int h, int k, int iters = 20, uint32_t seed = 42) {
    KMeansParams p{};
    p.width = w; p.height = h; p.k = k; p.max_iter = iters; p.seed = seed; p.num_threads = 1;
    return p;
}

static void test_determinism(ImageKind kind, int w, int h, int k) {
    const size_t n = static_cast<size_t>(w) * h;
    auto img = make_image(kind, w, h, 7);
    auto p = params(w, h, k);
    std::vector<int32_t> l1(n), l2(n);
    std::vector<uint8_t> c1(k * 3), c2(k * 3);
    const int r1 = ref_kmeans(img.data(), l1.data(), c1.data(), &p, nullptr);
    const int r2 = ref_kmeans(img.data(), l2.data(), c2.data(), &p, nullptr);
    char tag[128];
    std::snprintf(tag, sizeof(tag), "determinisme %s %dx%d K=%d", kind_name(kind), w, h, k);
    bool ok = (r1 == 0 && r2 == 0) && same_result(l1, c1, l2, c2, tag);
    for (size_t i = 0; ok && i < n; ++i) ok = (l1[i] >= 0 && l1[i] < k);
    check(ok, tag);
}

static void test_two_colors() {
    const int w = 40, h = 10, k = 2;
    const size_t n = static_cast<size_t>(w) * h;
    std::vector<uint8_t> img(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const bool left = (i % w) < w / 2;
        img[i * 3 + 0] = left ? 10 : 250;
        img[i * 3 + 1] = left ? 120 : 240;
        img[i * 3 + 2] = left ? 30 : 5;
    }
    auto p = params(w, h, k);
    std::vector<int32_t> lab(n);
    std::vector<uint8_t> cen(k * 3);
    bool ok = ref_kmeans(img.data(), lab.data(), cen.data(), &p, nullptr) == 0;
    std::set<int> got;
    for (int c = 0; c < k; ++c) got.insert(cen[c * 3] == 10 ? 0 : (cen[c * 3] == 250 ? 1 : 99));
    ok = ok && got == std::set<int>{0, 1};
    for (size_t i = 0; ok && i < n; ++i) {
        ok = (cen[lab[i] * 3 + 0] == img[i * 3 + 0] && cen[lab[i] * 3 + 1] == img[i * 3 + 1] &&
              cen[lab[i] * 3 + 2] == img[i * 3 + 2]);
    }
    check(ok, "dua warna K=2: centroid tepat dua warna, label sesuai");
}

static void test_edges() {
    // k > n: citra 1x1, K=16 tidak boleh macet dan semua label 0.
    std::vector<uint8_t> img = {1, 2, 3};
    auto p = params(1, 1, 16);
    std::vector<int32_t> lab(1, -1);
    std::vector<uint8_t> cen(16 * 3);
    bool ok = ref_kmeans(img.data(), lab.data(), cen.data(), &p, nullptr) == 0 && lab[0] == 0;
    for (int c = 0; ok && c < 16; ++c) ok = cen[c * 3] == 1 && cen[c * 3 + 1] == 2 && cen[c * 3 + 2] == 3;
    check(ok, "tepi: 1x1 dengan K=16");

    // Init: indeks berbeda bila n >= k (citra 4x4 dengan 16 warna unik, K=16).
    std::vector<uint8_t> uniq(16 * 3);
    for (int i = 0; i < 16; ++i) { uniq[i * 3] = static_cast<uint8_t>(i * 10); uniq[i * 3 + 1] = 0; uniq[i * 3 + 2] = 0; }
    std::vector<uint8_t> init(16 * 3);
    ok = kmeans_init_centroids(uniq.data(), 16, 16, 42, init.data()) == 0;
    std::set<int> reds;
    for (int c = 0; c < 16; ++c) reds.insert(init[c * 3]);
    check(ok && reds.size() == 16, "init: K=n memilih n piksel berbeda");

    // Argumen tidak sah harus ditolak (kembalian != 0).
    auto bad_k0 = params(4, 4, 0), bad_k17 = params(4, 4, 17), bad_it = params(4, 4, 2, 0),
         bad_w = params(0, 4, 2);
    bool rej = ref_kmeans(uniq.data(), lab.data(), cen.data(), &bad_k0, nullptr) != 0 &&
               ref_kmeans(uniq.data(), lab.data(), cen.data(), &bad_k17, nullptr) != 0 &&
               ref_kmeans(uniq.data(), lab.data(), cen.data(), &bad_it, nullptr) != 0 &&
               ref_kmeans(uniq.data(), lab.data(), cen.data(), &bad_w, nullptr) != 0 &&
               kmeans_init_centroids(nullptr, 16, 2, 42, init.data()) != 0 &&
               kmeans_init_centroids(uniq.data(), 0, 2, 42, init.data()) != 0;
    check(rej, "argumen tidak sah ditolak");
}

int main() {
    test_determinism(ImageKind::Random, 257, 131, 8);
    test_determinism(ImageKind::Gradient, 64, 64, 16);
    test_determinism(ImageKind::FewColors, 17, 13, 8);
    test_determinism(ImageKind::Solid, 33, 9, 4);
    test_two_colors();
    test_edges();

    // Cetak centroid satu kasus agar bisa dibandingkan antar mesin/kompilator.
    auto img = make_image(ImageKind::Random, 257, 131, 7);
    auto p = params(257, 131, 8);
    std::vector<int32_t> lab(257 * 131);
    std::vector<uint8_t> cen(8 * 3);
    StageTimes t{};
    ref_kmeans(img.data(), lab.data(), cen.data(), &p, &t);
    std::printf("Centroid acak 257x131 K=8 seed=42:");
    for (int c = 0; c < 8; ++c) std::printf(" (%u,%u,%u)", cen[c * 3], cen[c * 3 + 1], cen[c * 3 + 2]);
    std::printf("\n");

    std::printf("%s: %d kegagalan\n", g_fail ? "GAGAL" : "SEMUA LULUS", g_fail);
    return g_fail ? 1 : 0;
}
