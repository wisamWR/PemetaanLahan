// test_util.h - pembangkit citra uji dan pembanding hasil (khusus uji, bukan pustaka).
#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace testutil {

// PRNG uji (terpisah dari PRNG inisialisasi agar tidak saling memengaruhi).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

enum class ImageKind { Random, Solid, FewColors, Gradient };

inline const char* kind_name(ImageKind k) {
    switch (k) {
        case ImageKind::Random: return "acak";
        case ImageKind::Solid: return "satu_warna";
        case ImageKind::FewColors: return "3_warna";
        case ImageKind::Gradient: return "gradien";
    }
    return "?";
}

// Citra RGB interleaved 3*w*h byte.
inline std::vector<uint8_t> make_image(ImageKind kind, int w, int h, uint64_t seed) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<uint8_t> img(n * 3);
    Rng rng(seed);
    switch (kind) {
        case ImageKind::Random:
            for (size_t i = 0; i < n * 3; ++i) img[i] = static_cast<uint8_t>(rng.next() & 0xFF);
            break;
        case ImageKind::Solid:
            for (size_t i = 0; i < n; ++i) {
                img[i * 3 + 0] = 34; img[i * 3 + 1] = 139; img[i * 3 + 2] = 34;
            }
            break;
        case ImageKind::FewColors: {
            // 3 warna unik: memicu cluster kosong bila K > 3.
            static const uint8_t pal[3][3] = {{20, 90, 30}, {200, 180, 120}, {30, 60, 200}};
            for (size_t i = 0; i < n; ++i) {
                const uint64_t c = rng.next() % 3;
                img[i * 3 + 0] = pal[c][0]; img[i * 3 + 1] = pal[c][1]; img[i * 3 + 2] = pal[c][2];
            }
            break;
        }
        case ImageKind::Gradient:
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + x;
                    img[i * 3 + 0] = static_cast<uint8_t>(w > 1 ? (255LL * x) / (w - 1) : 0);
                    img[i * 3 + 1] = static_cast<uint8_t>(h > 1 ? (255LL * y) / (h - 1) : 0);
                    img[i * 3 + 2] = static_cast<uint8_t>((x + y) & 0xFF);
                }
            }
            break;
    }
    return img;
}

// Bandingkan hasil bit demi bit. Kembalikan true bila identik; cetak selisih pertama bila tidak.
inline bool same_result(const std::vector<int32_t>& la, const std::vector<uint8_t>& ca,
                        const std::vector<int32_t>& lb, const std::vector<uint8_t>& cb,
                        const std::string& tag) {
    if (ca.size() != cb.size() || std::memcmp(ca.data(), cb.data(), ca.size()) != 0) {
        for (size_t i = 0; i < ca.size(); ++i) {
            if (ca[i] != cb[i]) {
                std::printf("  [%s] centroid beda: cluster %zu kanal %zu: %u vs %u\n", tag.c_str(),
                            i / 3, i % 3, ca[i], cb[i]);
                break;
            }
        }
        return false;
    }
    if (la.size() != lb.size()) {
        std::printf("  [%s] jumlah label beda: %zu vs %zu\n", tag.c_str(), la.size(), lb.size());
        return false;
    }
    if (std::memcmp(la.data(), lb.data(), la.size() * sizeof(int32_t)) != 0) {
        size_t diff = 0, first = la.size();
        for (size_t i = 0; i < la.size(); ++i) {
            if (la[i] != lb[i]) { if (first == la.size()) first = i; ++diff; }
        }
        std::printf("  [%s] label beda di %zu piksel, pertama di indeks %zu: %d vs %d\n",
                    tag.c_str(), diff, first, la[first], lb[first]);
        return false;
    }
    return true;
}

}  // namespace testutil

#endif  // TEST_UTIL_H
