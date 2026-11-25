// probe_cuda.cu - percobaan pengukuran untuk analisis bottleneck (T4), bukan bagian pustaka.
//
// Mengukur dengan cudaEvent pada citra acak yang sama:
//   1. assign saja (hitung label, tulis label, TANPA akumulasi)  -> batas bawah biaya assign
//   2. kmeans_cuda_simple (T3, atomicAdd global)                 -> per iterasi
//   3. kmeans_cuda (T4, reduksi shared memory)                   -> per iterasi
//   4. salin device-ke-device sebesar lalu lintas memori satu iterasi assign (baca 3n + tulis 4n byte)
//      -> perkiraan waktu bila kernel murni terikat memori (bandwidth terukur, bukan spesifikasi)
//
//   probe_cuda [--width W --height H --k K --iters I --reps R]
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "kmeans.h"
#include "kmeans_cuda_variants.h"
#include "test_util.h"

namespace {

__global__ void assign_only(const uint8_t* __restrict__ rgb, size_t n,
                            const uint8_t* __restrict__ centroids, int k,
                            int32_t* __restrict__ labels) {
    __shared__ int s_cent[KMEANS_MAX_K * 3];
    for (int i = threadIdx.x; i < k * 3; i += blockDim.x) s_cent[i] = centroids[i];
    __syncthreads();
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int r = rgb[i * 3 + 0], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
        int best = 0, best_d = INT_MAX;
        for (int c = 0; c < k; ++c) {
            const int dr = r - s_cent[c * 3], dg = g - s_cent[c * 3 + 1], db = b - s_cent[c * 3 + 2];
            const int d = dr * dr + dg * dg + db * db;
            if (d < best_d) { best_d = d; best = c; }
        }
        labels[i] = best;
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int run(int w, int h, int k, int iters, int reps) {
    const size_t n = static_cast<size_t>(w) * h;
    auto img = testutil::make_image(testutil::ImageKind::Random, w, h, 1234);
    uint8_t cent[KMEANS_MAX_K * 3];
    if (kmeans_init_centroids(img.data(), n, k, 42, cent) != 0) return 1;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SM, L2 %d KB\n", prop.name, prop.multiProcessorCount,
                prop.l2CacheSize / 1024);
    std::printf("Citra acak %dx%d (%zu piksel), K=%d, %d iterasi, median %d kali\n\n", w, h, n, k,
                iters, reps);

    DeviceBuffer<uint8_t> d_rgb, d_cent, d_copy_src, d_copy_dst;
    DeviceBuffer<int32_t> d_lab;
    CUDA_CHECK(d_rgb.alloc(n * 3));
    CUDA_CHECK(d_lab.alloc(n));
    CUDA_CHECK(d_cent.alloc(KMEANS_MAX_K * 3));
    CUDA_CHECK(cudaMemcpy(d_rgb.ptr, img.data(), n * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent.ptr, cent, k * 3, cudaMemcpyHostToDevice));

    DeviceEvent e0, e1;
    CUDA_CHECK(e0.create());
    CUDA_CHECK(e1.create());
    const unsigned int blocks = static_cast<unsigned int>((n + 255) / 256);

    // 1. assign saja
    std::vector<double> t_assign;
    for (int r = 0; r <= reps; ++r) {
        CUDA_CHECK(cudaEventRecord(e0.ev));
        for (int it = 0; it < iters; ++it) {
            assign_only<<<blocks, 256>>>(d_rgb.ptr, n, d_cent.ptr, k, d_lab.ptr);
            CUDA_CHECK_LAUNCH();
        }
        CUDA_CHECK(cudaEventRecord(e1.ev));
        CUDA_CHECK(cudaEventSynchronize(e1.ev));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0.ev, e1.ev));
        if (r > 0) t_assign.push_back(ms / iters);
    }

    // 4. salin D2D sebesar lalu lintas satu iterasi (baca 3n, tulis 4n): cudaMemcpy D2D sebesar
    //    3,5n byte membaca 3,5n dan menulis 3,5n byte, total 7n byte.
    const size_t copy_bytes = n * 7 / 2;
    CUDA_CHECK(d_copy_src.alloc(copy_bytes));
    CUDA_CHECK(d_copy_dst.alloc(copy_bytes));
    CUDA_CHECK(cudaMemset(d_copy_src.ptr, 1, copy_bytes));
    std::vector<double> t_copy;
    for (int r = 0; r <= reps; ++r) {
        CUDA_CHECK(cudaEventRecord(e0.ev));
        CUDA_CHECK(cudaMemcpyAsync(d_copy_dst.ptr, d_copy_src.ptr, copy_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(e1.ev));
        CUDA_CHECK(cudaEventSynchronize(e1.ev));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0.ev, e1.ev));
        if (r > 0) t_copy.push_back(ms);
    }

    // 2 dan 3: lewat fungsi pustaka (kernel_ms mencakup seluruh loop iterasi)
    KMeansParams p{};
    p.width = w; p.height = h; p.k = k; p.max_iter = iters; p.seed = 42;
    std::vector<int32_t> labels(n);
    std::vector<uint8_t> out(k * 3);
    std::vector<double> t_simple, t_shared;
    for (int r = 0; r <= reps; ++r) {
        StageTimes t{};
        if (kmeans_cuda_simple(img.data(), labels.data(), out.data(), &p, &t) != 0) return 1;
        if (r > 0) t_simple.push_back(t.kernel_ms / iters);
        if (kmeans_cuda(img.data(), labels.data(), out.data(), &p, &t) != 0) return 1;
        if (r > 0) t_shared.push_back(t.kernel_ms / iters);
    }

    const double copy_ms = median(t_copy);
    std::printf("%-46s %10s\n", "pengukuran", "ms/iterasi");
    std::printf("%-46s %10.3f\n", "assign saja (tanpa akumulasi)", median(t_assign));
    std::printf("%-46s %10.3f\n", "T3 kmeans_cuda_simple (atomicAdd global)", median(t_simple));
    std::printf("%-46s %10.3f\n", "T4 kmeans_cuda (reduksi shared memory)", median(t_shared));
    std::printf("%-46s %10.3f  (%.1f GB/s efektif)\n", "salin D2D 7n byte (batas memori)", copy_ms,
                (7.0 * n) / (copy_ms * 1e6));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int w = 7680, h = 4320, k = 8, iters = 20, reps = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const int v = std::atoi(argv[i + 1]);
        if (a == "--width") w = v;
        else if (a == "--height") h = v;
        else if (a == "--k") k = v;
        else if (a == "--iters") iters = v;
        else if (a == "--reps") reps = v;
        else { std::fprintf(stderr, "argumen tidak dikenal: %s\n", a.c_str()); return 2; }
    }
    return run(w, h, k, iters, reps);
}
