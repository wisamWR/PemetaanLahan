// Membandingkan kmeans_cpu (Azka) dengan ref_kmeans (Wisam), bit demi bit.
#include <cstdio>
#include <vector>
#include "kmeans.h"
#include "ref_kmeans.h"

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
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) diff += (la[i] != lb[i]);
    for (size_t i = 0; i < ca.size(); i++) diff += (ca[i] != cb[i]);
    std::printf("%s %dx%d K=%d seed=%u: selisih=%zu\n",
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
