// png_io.h - baca/tulis PNG untuk citra RGB uint8 interleaved (kontrak butir 1).
#ifndef PNG_IO_H
#define PNG_IO_H

#include <cstdint>

// Membaca PNG menjadi RGB uint8, 3 byte per piksel, baris demi baris.
// Hasil dialokasikan oleh stb; bebaskan dengan png_free().
// Kembalian 0 = sukses, selain 0 = galat.
int png_read_rgb(const char* path, uint8_t** rgb, int* width, int* height);
void png_free(uint8_t* rgb);

// Menulis RGB uint8 interleaved (3*width*height byte) ke PNG 8-bit.
// Kembalian 0 = sukses, selain 0 = galat.
int png_write_rgb(const char* path, const uint8_t* rgb, int width, int height);

#endif  // PNG_IO_H
