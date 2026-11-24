// test_png_io.cpp - uji round-trip baca/tulis PNG (harus identik piksel per piksel).
#include <cstdio>
#include <vector>
#include "png_io.h"

int main() {
    const int W = 37, H = 23;  // ukuran ganjil agar tidak kebetulan
    std::vector<uint8_t> src(3 * W * H);
    uint32_t s = 2026;
    for (auto& v : src) { s = s * 1664525u + 1013904223u; v = (uint8_t)(s >> 24); }

    const char* path = "/tmp/png_io_roundtrip.png";
    int fail = 0;

    int rw = png_write_rgb(path, src.data(), W, H);
    std::printf("tulis: %s\n", rw == 0 ? "OK" : "GAGAL");
    fail |= (rw != 0);

    uint8_t* back = nullptr;
    int w2 = 0, h2 = 0;
    int rr = png_read_rgb(path, &back, &w2, &h2);
    std::printf("baca : %s\n", rr == 0 ? "OK" : "GAGAL");
    fail |= (rr != 0);

    if (rr == 0) {
        bool same = (w2 == W && h2 == H);
        for (size_t i = 0; same && i < src.size(); i++) same = (src[i] == back[i]);
        std::printf("round-trip identik: %s\n", same ? "OK" : "GAGAL");
        fail |= !same;
        png_free(back);
    }

    uint8_t* none = nullptr;
    int rn = png_read_rgb("/tmp/tidak_ada_file.png", &none, &w2, &h2);
    std::printf("file tidak ada ditolak: %s\n", rn != 0 ? "OK" : "GAGAL");
    fail |= (rn == 0);

    std::printf("%s\n", fail ? "HASIL: ADA YANG GAGAL" : "HASIL: SEMUA LULUS");
    return fail;
}
