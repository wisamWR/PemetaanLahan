// Mengukur waktu tulis PNG untuk backend png_io yang ditautkan (stb atau omp) pada citra peta-kelas.
//   make bench-png                      (backend omp, sapuan 1,2,4,8 thread)
//   ./build/bench_png_write 4000 3000   (ukuran lain)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "png_io.h"

int main(int argc, char** argv) {
    const int w = argc > 1 ? std::atoi(argv[1]) : 4000, h = argc > 2 ? std::atoi(argv[2]) : 3000;
    const int reps = 5;
    static const uint8_t pal[8][3] = {{30, 70, 160}, {40, 120, 50}, {170, 150, 100}, {150, 150, 150},
                                      {90, 160, 70}, {20, 20, 20}, {240, 240, 240}, {200, 30, 30}};
    std::vector<uint8_t> img((size_t)w * h * 3);
    uint64_t s = 88172645463325252ULL;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            int r = ((x / 97) * 7 + (y / 131) * 13 + (x / 311) * (y / 277)) % 8;
            if (s % 50 == 0) r = (int)((s >> 8) % 8);
            for (int c = 0; c < 3; ++c) img[((size_t)y * w + x) * 3 + c] = pal[r][c];
        }
    const std::string out = "build/bench_png_write.png";
    int maxt = 1;
#ifdef _OPENMP
    maxt = omp_get_max_threads();
#endif
    std::printf("peta-kelas %dx%d (%.1f MB RGB), median dari %d, tulis 1 warm-up dibuang\n", w, h, img.size() / 1e6, reps);
    std::printf("%8s %10s %10s\n", "thread", "tulis(ms)", "ukuran(MB)");
    for (int th = 1; th <= maxt; th *= 2) {
#ifdef _OPENMP
        omp_set_num_threads(th);
#endif
        std::vector<double> ms;
        std::string err;
        for (int r = 0; r <= reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            if (!lc::png_write_rgb8(out, img.data(), w, h, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
            double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r > 0) ms.push_back(d);
        }
        std::sort(ms.begin(), ms.end());
        struct stat st; stat(out.c_str(), &st);
        std::printf("%8d %10.1f %10.2f\n", th, ms[ms.size() / 2], st.st_size / 1e6);
        if (th * 2 > maxt && th != maxt) { th = maxt / 2; }   // pastikan maxt ikut diukur
    }
    return 0;
}
