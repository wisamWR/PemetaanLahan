// png_io_stb.cpp - implementasi png_io.h dengan stb_image dan stb_image_write.
//
// Memakai third_party/stb/stb_image.h (v2.34) dan third_party/stb/stb_image_write.h (v1.19)
// dari github.com/nothings/stb, commit 6b5799bae8a1fe3792691ef4b72ae26434c3e799 (domain
// publik/MIT; ukuran 278.442 dan 80.947 byte). Logika program juga bisa diuji dengan backend
// PNG pengganti (make landcover LC_PNG_SRC=...).
//
// Catatan kinerja (terukur 22 Nov di WSL, tile 10980x10980 RGB, derau acak = kasus terburuk
// untuk kompresi): stbi_write_png membuat salinan tersaring seluruh citra di memori (±362 MB)
// lalu memampatkannya dengan deflate sederhana buatan stb. map.png 8 warna dari tile penuh
// ±4,4 s dan 25,7 MB; membaca satu PNG 377 MB ±7,1 s serial (16 potongan, 8 thread: ±2,0 s).
// Filter dipaksa tetap (bukan pilihan per baris) supaya tahap "tulis" tidak 5x lebih lambat;
// ubah LC_PNG_FILTER bila pengukuran menyarankan lain.

#include "png_io.h"

#include <cstdio>

#ifndef LC_PNG_FILTER
#define LC_PNG_FILTER 1  // 0 none, 1 sub, 2 up, 3 average, 4 paeth, -1 pilih per baris (lambat)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace lc {

namespace {

std::string reason() {
    const char* r = stbi_failure_reason();
    return r ? r : "alasan tidak diketahui";
}

}  // namespace

bool png_info(const std::string& path, int& width, int& height, std::string& err) {
    int comp = 0;
    if (!stbi_info(path.c_str(), &width, &height, &comp)) {
        err = path + ": bukan PNG yang bisa dibaca (" + reason() + ")";
        return false;
    }
    if (width < 1 || height < 1) {
        err = path + ": ukuran PNG tidak sah";
        return false;
    }
    return true;
}

bool png_read_rgb8(const std::string& path, Rgb8& out, std::string& err) {
    int w = 0, h = 0, comp = 0;
    uint8_t* px = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (px == nullptr) {
        err = path + ": gagal membaca PNG (" + reason() + ")";
        return false;
    }
    out.width = w;
    out.height = h;
    out.data.reset(px);  // stb memakai malloc, jadi std::free benar
    return true;
}

namespace {

bool write_png(const std::string& path, const uint8_t* px, int w, int h, int comp,
               std::string& err) {
    if (w < 1 || h < 1 || px == nullptr) {
        err = path + ": ukuran atau data PNG tidak sah";
        return false;
    }
    stbi_write_force_png_filter = LC_PNG_FILTER;
    if (!stbi_write_png(path.c_str(), w, h, comp, px, w * comp)) {
        err = path + ": gagal menulis PNG";
        return false;
    }
    return true;
}

}  // namespace

bool png_write_gray8(const std::string& path, const uint8_t* px, int width, int height,
                     std::string& err) {
    return write_png(path, px, width, height, 1, err);
}

bool png_write_rgb8(const std::string& path, const uint8_t* px, int width, int height,
                    std::string& err) {
    return write_png(path, px, width, height, 3, err);
}

}  // namespace lc
