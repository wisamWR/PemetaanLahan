// png_io.h - antarmuka baca/tulis PNG untuk program landcover (Ridlo, Langkah 11a).
//
// Sengaja kecil supaya implementasinya bisa diganti tanpa menyentuh landcover.cpp:
//   - src/cli/png_io_stb.cpp : stb_image + stb_image_write (third_party/stb/, butuh izin unduh)
//   - fungsi baca/tulis PNG Azka, bila nanti ada dan disepakati
//   - libpng/zlib sistem (butuh zlib1g-dev, libpng-dev)
//
// Semua fungsi mengembalikan true bila sukses. Bila gagal: false dan `err` berisi pesan
// (bahasa Indonesia) yang boleh langsung ditampilkan; tidak ada pengecualian yang keluar,
// jadi aman dipanggil dari daerah paralel OpenMP.
#ifndef LANDCOVER_PNG_IO_H
#define LANDCOVER_PNG_IO_H

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace lc {

// Citra RGB uint8 interleaved (3 * width * height byte), dialokasikan dengan malloc.
struct Rgb8 {
    int width = 0;
    int height = 0;
    std::unique_ptr<uint8_t, void (*)(void*)> data{nullptr, std::free};
};

// Hanya membaca header (tanpa dekode piksel).
bool png_info(const std::string& path, int& width, int& height, std::string& err);

// Dekode ke RGB 8-bit. Kanal alfa dibuang, abu-abu dan palet diperluas ke RGB, 16-bit
// dipotong ke 8-bit (stb: byte tinggi).
bool png_read_rgb8(const std::string& path, Rgb8& out, std::string& err);

// Tulis PNG 8-bit grayscale (w*h byte) atau RGB (3*w*h byte) ke `path` apa adanya
// (pemanggil yang menulis ke berkas .tmp lalu me-rename).
bool png_write_gray8(const std::string& path, const uint8_t* px, int width, int height,
                     std::string& err);
bool png_write_rgb8(const std::string& path, const uint8_t* px, int width, int height,
                    std::string& err);

}  // namespace lc

#endif  // LANDCOVER_PNG_IO_H
