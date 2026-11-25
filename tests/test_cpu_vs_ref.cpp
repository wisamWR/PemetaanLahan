// Membandingkan kmeans_cpu (Azka) dengan ref_kmeans (Wisam), bit demi bit, memakai verify_identical.
#include <cstdio>
#include <vector>
#include "kmeans.h"
#include "ref_kmeans.h"
#include "verify.h"

static std::vector<uint8_t> make_image(int w, int h, uint32_t seed) {
    std::vector<uint8_t> img(3 * (size_t)w * h);
    uint32_t s = seed;
    for (auto& v : img) {
        s = s * 1664525u + 1013904223u;
        v = (uint8_t)(s >> 24);
    }
    return img;
}

static int compare(int w, int h, int k, uint32_t seed) {
    auto img = make_image(w, h, seed);
    size_t n = (size_t)w * h;
    KMeansParams p{w, h, k, 20, 42, 1};
    std::vector<int32_t> la(n), lb(n);
    std::vector<uint8_t> ca(3 * k), cb(3 * k);
    int ra = kmeans_cpu(img.data(), la.data(), ca.data(), &p, nullptr);
    int rb = ref_kmeans(img.data(), lb.data(), cb.data(), &p, nullptr);
    if (ra || rb) { std::printf("GAGAL panggilan: %d %d\n", ra, rb); return 1; }
    long diff = verify_identical(lb.data(), cb.data(), la.data(), ca.data(), n, k, stdout);
    std::printf("%s %dx%d K=%d seed=%u: selisih=%ld\n",
                diff == 0 ? "[SAMA]" : "[BEDA]", w, h, k, seed, diff);
    return diff != 0;
}

int main() {
    int fail = 0;
    fail |= compare(257, 131, 8, 7);
    fail |= compare(64, 64, 16, 3);
    fail |= compare(17, 13, 8, 11);
    fail |= compare(1, 1, 1, 1);
    std::printf("%s\n", fail ? "HASIL: ADA YANG BEDA" : "HASIL: SEMUA SAMA");
    return fail;
}
