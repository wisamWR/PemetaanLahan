/* K-Means OpenMP (L2). Hasil harus identik bit demi bit dengan kmeans_cpu (L0).
 *
 * Mengapa identik walau paralel: jumlah per cluster memakai bilangan bulat (uint64_t),
 * sehingga hasil penjumlahan tidak bergantung urutan penambahan. Tie-break memakai
 * perbandingan "<" ketat di loop c = 0..k-1 (indeks terkecil), sama dengan L0.
 */
#include <cstring>
#include <omp.h>
#include "kmeans.h"

int kmeans_omp(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
               const KMeansParams* p, StageTimes* t) {
    /* Validasi sama dengan kmeans_cpu (U4). */
    if (!rgb || !labels || !centroids || !p) return 1;
    if (p->width < 1 || p->height < 1) return 2;
    if (p->k < 1 || p->k > KMEANS_MAX_K) return 3;
    if (p->max_iter < 1) return 4;

    const long long n = (long long)p->width * (long long)p->height;
    const int k = p->k;
    const int nt = p->num_threads > 0 ? p->num_threads : omp_get_max_threads();

    double ti = omp_get_wtime();
    if (kmeans_init_centroids(rgb, (size_t)n, k, p->seed, centroids)) return 5;
    double init_ms = (omp_get_wtime() - ti) * 1000.0;

    uint64_t sum[KMEANS_MAX_K * 3];
    uint64_t count[KMEANS_MAX_K];

    double tk = omp_get_wtime();
    for (int iter = 0; iter < p->max_iter; iter++) {
        std::memset(sum, 0, sizeof sum);
        std::memset(count, 0, sizeof count);

        #pragma omp parallel num_threads(nt)
        {
            /* Akumulator lokal per thread (di stack, menghindari false sharing). */
            uint64_t lsum[KMEANS_MAX_K * 3] = {0};
            uint64_t lcnt[KMEANS_MAX_K] = {0};
            uint8_t  cen[KMEANS_MAX_K * 3];
            std::memcpy(cen, centroids, (size_t)k * 3);

            #pragma omp for schedule(static) nowait
            for (long long i = 0; i < n; i++) {
                const uint8_t* px = rgb + 3 * i;
                int best = 0;
                int32_t bestd = INT32_MAX;
                for (int c = 0; c < k; c++) {
                    int32_t dr = (int32_t)px[0] - cen[3 * c + 0];
                    int32_t dg = (int32_t)px[1] - cen[3 * c + 1];
                    int32_t db = (int32_t)px[2] - cen[3 * c + 2];
                    int32_t d = dr * dr + dg * dg + db * db;
                    if (d < bestd) { bestd = d; best = c; }
                }
                labels[i] = best;
                lsum[3 * best + 0] += px[0];
                lsum[3 * best + 1] += px[1];
                lsum[3 * best + 2] += px[2];
                lcnt[best]++;
            }

            #pragma omp critical
            {
                for (int j = 0; j < k * 3; j++) sum[j] += lsum[j];
                for (int c = 0; c < k; c++) count[c] += lcnt[c];
            }
        }

        for (int c = 0; c < k; c++) {
            if (count[c] == 0) continue;   /* cluster kosong: tetap */
            for (int ch = 0; ch < 3; ch++)
                centroids[3 * c + ch] = (uint8_t)(sum[3 * c + ch] / count[c]);
        }
    }
    double kmeans_ms = (omp_get_wtime() - tk) * 1000.0;

    if (t) {
        std::memset(t, 0, sizeof *t);
        t->init_ms = init_ms;
        t->kmeans_ms = kmeans_ms;
    }
    return 0;
}
