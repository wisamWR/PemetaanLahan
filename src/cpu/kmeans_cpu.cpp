/* K-Means serial (L0, acuan). Mengikuti kontrak butir 3 dan U1-U5. */
#include <chrono>
#include <cstring>
#include "kmeans.h"

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int kmeans_cpu(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
               const KMeansParams* p, StageTimes* t) {
    /* Validasi U4. */
    if (!rgb || !labels || !centroids || !p) return 1;
    if (p->width < 1 || p->height < 1) return 2;
    if (p->k < 1 || p->k > KMEANS_MAX_K) return 3;
    if (p->max_iter < 1) return 4;

    const size_t n = (size_t)p->width * (size_t)p->height;
    const int k = p->k;

    /* Inisialisasi bersama (butir 3.1). */
    double ti = now_ms();
    if (kmeans_init_centroids(rgb, n, k, p->seed, centroids)) return 5;
    double init_ms = now_ms() - ti;

    uint64_t sum[KMEANS_MAX_K * 3];
    uint64_t count[KMEANS_MAX_K];

    double tk = now_ms();
    for (int iter = 0; iter < p->max_iter; iter++) {
        /* Assign terhadap centroid iterasi ini (butir 3.7). */
        std::memset(sum, 0, sizeof sum);
        std::memset(count, 0, sizeof count);
        for (size_t i = 0; i < n; i++) {
            const uint8_t* px = rgb + 3 * i;
            int best = 0;
            int32_t bestd = INT32_MAX;
            for (int c = 0; c < k; c++) {
                const uint8_t* ce = centroids + 3 * c;
                int32_t dr = (int32_t)px[0] - ce[0];
                int32_t dg = (int32_t)px[1] - ce[1];
                int32_t db = (int32_t)px[2] - ce[2];
                int32_t d = dr * dr + dg * dg + db * db;   /* butir 3.2 */
                if (d < bestd) {                           /* strict: seri -> indeks terkecil (3.3) */
                    bestd = d;
                    best = c;
                }
            }
            labels[i] = best;
            sum[3 * best + 0] += px[0];
            sum[3 * best + 1] += px[1];
            sum[3 * best + 2] += px[2];
            count[best]++;
        }
        /* Update centroid (butir 3.4, 3.5). */
        for (int c = 0; c < k; c++) {
            if (count[c] == 0) continue;   /* cluster kosong: tetap */
            for (int ch = 0; ch < 3; ch++)
                centroids[3 * c + ch] = (uint8_t)(sum[3 * c + ch] / count[c]);
        }
    }
    double kmeans_ms = now_ms() - tk;

    if (t) {
        std::memset(t, 0, sizeof *t);
        t->init_ms = init_ms;
        t->kmeans_ms = kmeans_ms;
    }
    return 0;
}
