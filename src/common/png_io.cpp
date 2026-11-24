// png_io.cpp - implementasi baca/tulis PNG memakai stb_image dan stb_image_write.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "png_io.h"

int png_read_rgb(const char* path, uint8_t** rgb, int* width, int* height) {
    if (!path || !rgb || !width || !height) return 1;
    int w = 0, h = 0, c = 0;
    unsigned char* px = stbi_load(path, &w, &h, &c, 3);  // paksa 3 kanal RGB
    if (!px) return 2;
    *rgb = px;
    *width = w;
    *height = h;
    return 0;
}

void png_free(uint8_t* rgb) { stbi_image_free(rgb); }

int png_write_rgb(const char* path, const uint8_t* rgb, int width, int height) {
    if (!path || !rgb || width < 1 || height < 1) return 1;
    return stbi_write_png(path, width, height, 3, rgb, width * 3) ? 0 : 2;
}
