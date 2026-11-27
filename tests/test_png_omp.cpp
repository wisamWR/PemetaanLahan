// Uji penulis PNG paralel (src/cli/png_io_omp.cpp): tulis lalu baca kembali dengan stb_image,
// piksel harus sama persis; berkas harus identik byte untuk berapa pun jumlah thread.
//   make test-png-omp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "png_io.h"

namespace {

enum Kind { Noise, Flat, Gradient, ClassMap, Sub };
const char* kind_name(int k) {
    static const char* n[] = {"derau", "rata", "gradien", "peta-kelas", "tepi-ekstrem"};
    return n[k];
}

std::vector<uint8_t> make(int kind, int w, int h, int comp) {
    std::vector<uint8_t> v(static_cast<size_t>(w) * h * comp);
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ (uint64_t)w * 131 ^ (uint64_t)h * 7919 ^ (uint64_t)kind;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    static const uint8_t pal[8][3] = {{30, 70, 160}, {40, 120, 50}, {170, 150, 100}, {150, 150, 150},
                                      {90, 160, 70}, {0, 0, 0}, {255, 255, 255}, {200, 30, 30}};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t px[3];
            switch (kind) {
                case Noise: for (int c = 0; c < 3; ++c) px[c] = (uint8_t)rnd(); break;
                case Flat: px[0] = 12; px[1] = 200; px[2] = 77; break;
                case Gradient: px[0] = (uint8_t)(x * 255 / (w > 1 ? w - 1 : 1)); px[1] = (uint8_t)(y * 255 / (h > 1 ? h - 1 : 1)); px[2] = (uint8_t)(x + y); break;
                case ClassMap: {
                    int r = ((x / 97) * 7 + (y / 131) * 13 + (x / 311) * (y / 277)) % 8;
                    if (rnd() % 50 == 0) r = (int)(rnd() % 8);   // piksel tersebar di batas kelas
                    std::memcpy(px, pal[r], 3);
                    break;
                }
                default: { uint8_t b = (rnd() & 1) ? 255 : 0; px[0] = b; px[1] = (uint8_t)~b; px[2] = b; }
            }
            for (int c = 0; c < comp; ++c) v[((size_t)y * w + x) * comp + c] = px[c];
        }
    return v;
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

int failures = 0;
void check(bool ok, const char* what, int kind, int w, int h, int comp, int th) {
    if (!ok) {
        ++failures;
        std::printf("[GAGAL] %s  jenis=%s %dx%d kanal=%d thread=%d\n", what, kind_name(kind), w, h, comp, th);
    }
}

}  // namespace

int main() {
    const std::string tmp = "build/test_png_omp.png";
    const int sizes[][2] = {{1, 1}, {1, 700}, {700, 1}, {2, 3}, {17, 13}, {257, 131}, {1000, 700}, {3001, 1777}};
    const int threads[] = {1, 2, 3, 8};
    int cases = 0;
    for (int kind = Noise; kind <= Sub; ++kind)
        for (auto& sz : sizes)
            for (int comp : {1, 3}) {
                const int w = sz[0], h = sz[1];
                auto img = make(kind, w, h, comp);
                std::string first;
                for (int th : threads) {
#ifdef _OPENMP
                    omp_set_num_threads(th);
#endif
                    std::string err;
                    const bool wok = comp == 3 ? lc::png_write_rgb8(tmp, img.data(), w, h, err)
                                               : lc::png_write_gray8(tmp, img.data(), w, h, err);
                    check(wok, "tulis gagal", kind, w, h, comp, th);
                    if (!wok) continue;
                    std::string bytes = slurp(tmp);
                    if (th == threads[0]) first = bytes;
                    else check(bytes == first, "berkas beda antar jumlah thread", kind, w, h, comp, th);
                    // baca kembali dengan stb (RGB); gray diperluas ke RGB oleh png_read_rgb8
                    lc::Rgb8 back;
                    const bool rok = lc::png_read_rgb8(tmp, back, err);
                    check(rok && back.width == w && back.height == h, "baca ulang gagal / ukuran beda", kind, w, h, comp, th);
                    if (rok) {
                        bool same = true;
                        for (size_t i = 0; i < (size_t)w * h && same; ++i)
                            for (int c = 0; c < 3; ++c)
                                if (back.data.get()[i * 3 + c] != img[i * comp + (comp == 3 ? c : 0)]) { same = false; break; }
                        check(same, "piksel hasil baca beda dari masukan", kind, w, h, comp, th);
                    }
                    ++cases;
                }
            }
    std::printf("%s (%d kasus)\n", failures == 0 ? "HASIL: SEMUA LOLOS" : "HASIL: ADA YANG GAGAL", cases);
    return failures == 0 ? 0 : 1;
}
