// ref_kmeans.cpp - referensi CPU sementara, ditulis selurus mungkin mengikuti kontrak
// (docs/KONTRAK_TEKNIS.md butir 3). Tujuannya kebenaran, bukan kecepatan.
//
// Per iterasi (tepat max_iter kali, tanpa penghentian dini):
//   1. assign: tiap piksel ke centroid dengan jarak kuadrat int32 terkecil;
//      tie-break ke indeks cluster terkecil (perbandingan '<' ketat).
//   2. update: centroid[c][ch] = sum[c][ch] / count[c] (pembagian bilangan bulat, uint64);
//      cluster kosong (count == 0) tidak berubah.
// Label keluaran = hasil assign iterasi terakhir; centroid keluaran = setelah update terakhir.

#include "ref_kmeans.h"

#include <chrono>
#include <cstring>

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

extern "C" int ref_kmeans(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                          const KMeansParams* p, StageTimes* t) {
    if (rgb == nullptr || labels == nullptr || centroids == nullptr || p == nullptr) return 1;
    if (p->width < 1 || p->height < 1) return 1;
    if (p->k < 1 || p->k > KMEANS_MAX_K) return 1;
    if (p->max_iter < 1) return 1;

    const auto t_start = std::chrono::steady_clock::now();
    const size_t n = static_cast<size_t>(p->width) * static_cast<size_t>(p->height);
    const int k = p->k;

    if (kmeans_init_centroids(rgb, n, k, p->seed, centroids) != 0) return 1;
    const double init_ms = ms_since(t_start);

    const auto t_km = std::chrono::steady_clock::now();
    uint64_t sums[KMEANS_MAX_K * 3];
    uint64_t counts[KMEANS_MAX_K];

    for (int it = 0; it < p->max_iter; ++it) {
        std::memset(sums, 0, sizeof(sums));
        std::memset(counts, 0, sizeof(counts));

        for (size_t i = 0; i < n; ++i) {
            const int32_t r = rgb[i * 3 + 0];
            const int32_t g = rgb[i * 3 + 1];
            const int32_t b = rgb[i * 3 + 2];
            int32_t best = 0;
            int32_t best_d = 0x7FFFFFFF;
            for (int c = 0; c < k; ++c) {
                const int32_t dr = r - centroids[c * 3 + 0];
                const int32_t dg = g - centroids[c * 3 + 1];
                const int32_t db = b - centroids[c * 3 + 2];
                const int32_t d = dr * dr + dg * dg + db * db;
                if (d < best_d) { best_d = d; best = c; }
            }
            labels[i] = best;
            sums[best * 3 + 0] += static_cast<uint64_t>(r);
            sums[best * 3 + 1] += static_cast<uint64_t>(g);
            sums[best * 3 + 2] += static_cast<uint64_t>(b);
            counts[best] += 1;
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;  // cluster kosong: centroid tetap
            for (int ch = 0; ch < 3; ++ch) {
                centroids[c * 3 + ch] = static_cast<uint8_t>(sums[c * 3 + ch] / counts[c]);
            }
        }
    }

    if (t != nullptr) {
        std::memset(t, 0, sizeof(*t));
        t->init_ms = init_ms;
        t->kmeans_ms = ms_since(t_km);
        t->total_ms = ms_since(t_start);
    }
    return 0;
}
