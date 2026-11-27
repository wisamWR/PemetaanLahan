// Membandingkan kmeans_omp (Gavra) dengan kmeans_cpu (Azka), bit demi bit,
// untuk beberapa jenis citra, nilai K, dan jumlah thread.
#include <cstdio>
#include <vector>
#include "kmeans.h"
#include "test_util.h"

using namespace testutil;

static bool run(ImageKind kind, int w, int h, int k, int threads) {
    auto img = make_image(kind, w, h, 123);
    size_t n = (size_t)w * h;
    KMeansParams p{w, h, k, 20, 42, threads};
    std::vector<int32_t> la(n), lb(n);
    std::vector<uint8_t> ca(3 * k), cb(3 * k);
    int ra = kmeans_cpu(img.data(), la.data(), ca.data(), &p, nullptr);
    int rb = kmeans_omp(img.data(), lb.data(), cb.data(), &p, nullptr);
    if (ra || rb) { std::printf("[GAGAL] panggilan: %d %d\n", ra, rb); return false; }
    bool ok = same_result(la, ca, lb, cb, "omp");
    std::printf("%s %-10s %dx%d K=%d thread=%d\n", ok ? "[SAMA]" : "[BEDA]",
                kind_name(kind), w, h, k, threads);
    return ok;
}

int main() {
    bool ok = true;
    const ImageKind kinds[] = {ImageKind::Random, ImageKind::Solid, ImageKind::FewColors, ImageKind::Gradient};
    const int sizes[][2] = {{257, 131}, {64, 64}, {17, 13}, {1, 1}, {1000, 700}};
    const int ks[] = {1, 4, 8, 16};
    const int ths[] = {1, 2, 3, 4, 8};
    for (auto kind : kinds)
        for (auto& s : sizes)
            for (int k : ks)
                for (int th : ths)
                    ok &= run(kind, s[0], s[1], k, th);
    std::printf("%s\n", ok ? "HASIL: SEMUA SAMA" : "HASIL: ADA YANG BEDA");
    return ok ? 0 : 1;
}
