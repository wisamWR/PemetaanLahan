// kmeans_cuda.cu - K-Means L1 versi sederhana (T3): akumulasi atomicAdd pada memori global.
//
// Alur (mengikuti kontrak butir 3, harus identik dengan referensi CPU):
//   host : validasi argumen, kmeans_init_centroids (src/common), salin citra + centroid ke device
//   loop tepat max_iter kali, seluruhnya di device (centroid tidak bolak-balik ke host):
//     1. cudaMemset sums/counts = 0
//     2. assign_accumulate: satu thread per piksel (grid-stride), jarak kuadrat int32 ke K centroid,
//        tie-break indeks terkecil, tulis label int32, atomicAdd 64-bit ke sums[K*3] dan counts[K]
//     3. update_centroids: satu blok kecil, centroid = sum / count (pembagian bilangan bulat),
//        cluster kosong tidak berubah
//   host : salin label dan centroid ke host
//
// Waktu (cudaEvent): h2d_ms = salin citra + centroid, kernel_ms = seluruh loop iterasi,
// d2h_ms = salin label + centroid. kmeans_ms = h2d + loop + d2h. init_ms dan total_ms dari jam host
// (total_ms termasuk alokasi dan pembebasan memori device).

#include <chrono>
#include <climits>
#include <cstring>

#include "cuda_utils.h"
#include "kmeans.h"

namespace {

constexpr int kBlockSize = 256;

// Satu thread per piksel dengan grid-stride loop. Indeks memakai size_t (n bisa 120 juta,
// offset RGB 3*i sampai sekitar 360 juta; tetap dihitung 64-bit agar aman).
__global__ void assign_accumulate(const uint8_t* __restrict__ rgb, size_t n,
                                  const uint8_t* __restrict__ centroids, int k,
                                  int32_t* __restrict__ labels,
                                  unsigned long long* __restrict__ sums,
                                  unsigned long long* __restrict__ counts) {
    __shared__ int s_cent[KMEANS_MAX_K * 3];
    for (int i = threadIdx.x; i < k * 3; i += blockDim.x) s_cent[i] = centroids[i];
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const size_t off = i * 3;
        const int r = rgb[off + 0];
        const int g = rgb[off + 1];
        const int b = rgb[off + 2];

        int best = 0;
        int best_d = INT_MAX;
        for (int c = 0; c < k; ++c) {
            const int dr = r - s_cent[c * 3 + 0];
            const int dg = g - s_cent[c * 3 + 1];
            const int db = b - s_cent[c * 3 + 2];
            const int d = dr * dr + dg * dg + db * db;
            if (d < best_d) { best_d = d; best = c; }  // '<' ketat: tie-break indeks terkecil
        }
        labels[i] = best;
        atomicAdd(&sums[best * 3 + 0], static_cast<unsigned long long>(r));
        atomicAdd(&sums[best * 3 + 1], static_cast<unsigned long long>(g));
        atomicAdd(&sums[best * 3 + 2], static_cast<unsigned long long>(b));
        atomicAdd(&counts[best], 1ULL);
    }
}

__global__ void update_centroids(uint8_t* centroids, int k, const unsigned long long* sums,
                                 const unsigned long long* counts) {
    const int c = threadIdx.x;
    if (c >= k) return;
    const unsigned long long cnt = counts[c];
    if (cnt == 0) return;  // cluster kosong: centroid tetap
    for (int ch = 0; ch < 3; ++ch) {
        centroids[c * 3 + ch] = static_cast<uint8_t>(sums[c * 3 + ch] / cnt);
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int run(const uint8_t* rgb, int32_t* labels, uint8_t* centroids, const KMeansParams* p,
        StageTimes* t) {
    if (rgb == nullptr || labels == nullptr || centroids == nullptr || p == nullptr) return 1;
    if (p->width < 1 || p->height < 1) return 1;
    if (p->k < 1 || p->k > KMEANS_MAX_K) return 1;
    if (p->max_iter < 1) return 1;

    const auto t_start = std::chrono::steady_clock::now();
    const size_t n = static_cast<size_t>(p->width) * static_cast<size_t>(p->height);
    const int k = p->k;

    // Inisialisasi bersama di host (kontrak butir 3.1). Mode CUDA tidak punya inisialisasi sendiri.
    uint8_t init_cent[KMEANS_MAX_K * 3];
    if (kmeans_init_centroids(rgb, n, k, p->seed, init_cent) != 0) return 1;
    const double init_ms = ms_since(t_start);

    DeviceBuffer<uint8_t> d_rgb, d_cent;
    DeviceBuffer<int32_t> d_labels;
    DeviceBuffer<unsigned long long> d_sums, d_counts;
    CUDA_CHECK(d_rgb.alloc(n * 3));
    CUDA_CHECK(d_labels.alloc(n));
    CUDA_CHECK(d_cent.alloc(KMEANS_MAX_K * 3));
    CUDA_CHECK(d_sums.alloc(KMEANS_MAX_K * 3));
    CUDA_CHECK(d_counts.alloc(KMEANS_MAX_K));

    DeviceEvent e_h2d0, e_h2d1, e_loop1, e_d2h1;
    CUDA_CHECK(e_h2d0.create());
    CUDA_CHECK(e_h2d1.create());
    CUDA_CHECK(e_loop1.create());
    CUDA_CHECK(e_d2h1.create());

    // Ukuran grid: cukup untuk menutup semua piksel, dibatasi agar tetap dalam batas gridDim.x;
    // sisanya ditangani grid-stride loop.
    const size_t want_blocks = (n + kBlockSize - 1) / kBlockSize;
    const unsigned int blocks =
        static_cast<unsigned int>(want_blocks < (1u << 30) ? want_blocks : (1u << 30));

    CUDA_CHECK(cudaEventRecord(e_h2d0.ev));
    CUDA_CHECK(cudaMemcpy(d_rgb.ptr, rgb, n * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent.ptr, init_cent, static_cast<size_t>(k) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_h2d1.ev));

    for (int it = 0; it < p->max_iter; ++it) {
        CUDA_CHECK(cudaMemsetAsync(d_sums.ptr, 0, KMEANS_MAX_K * 3 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(d_counts.ptr, 0, KMEANS_MAX_K * sizeof(unsigned long long)));
        assign_accumulate<<<blocks, kBlockSize>>>(d_rgb.ptr, n, d_cent.ptr, k, d_labels.ptr,
                                                  d_sums.ptr, d_counts.ptr);
        CUDA_CHECK_LAUNCH();
        update_centroids<<<1, 32>>>(d_cent.ptr, k, d_sums.ptr, d_counts.ptr);
        CUDA_CHECK_LAUNCH();
    }
    CUDA_CHECK(cudaEventRecord(e_loop1.ev));

    CUDA_CHECK(cudaMemcpy(labels, d_labels.ptr, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(centroids, d_cent.ptr, static_cast<size_t>(k) * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h1.ev));
    CUDA_CHECK(cudaEventSynchronize(e_d2h1.ev));

    float h2d = 0.f, loop = 0.f, d2h = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, e_h2d0.ev, e_h2d1.ev));
    CUDA_CHECK(cudaEventElapsedTime(&loop, e_h2d1.ev, e_loop1.ev));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, e_loop1.ev, e_d2h1.ev));

    if (t != nullptr) {
        std::memset(t, 0, sizeof(*t));
        t->init_ms = init_ms;
        t->h2d_ms = h2d;
        t->kernel_ms = loop;
        t->d2h_ms = d2h;
        t->kmeans_ms = static_cast<double>(h2d) + loop + d2h;
    }
    // total_ms diisi setelah buffer device dibebaskan (lihat kmeans_cuda).
    return 0;
}

}  // namespace

extern "C" int kmeans_cuda(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                           const KMeansParams* p, StageTimes* t) {
    // Inisialisasi konteks CUDA di luar pengukuran (panggilan pertama bisa ratusan ms).
    CUDA_CHECK(cudaFree(nullptr));

    const auto t0 = std::chrono::steady_clock::now();
    const int rc = run(rgb, labels, centroids, p, t);  // buffer device dibebaskan saat run() selesai
    if (rc == 0 && t != nullptr) t->total_ms = ms_since(t0);
    return rc;
}
