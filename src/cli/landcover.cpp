// landcover.cpp - program penghubung (Ridlo, Langkah 11a).
//
// Membaca citra (satu PNG, atau folder dengan manifest.json + potongan PNG), menjalankan
// K-Means sesuai --mode/--kmeans, menghitung counts dan pratinjau dari label, lalu menulis
// folder run (meta.json, timing.json, centroids.json, counts.json, labels_preview.png,
// map_preview.png, map.png, labels.u8 opsional) yang dibaca UI lewat app/landcover/contract.py;
// pembaca itulah definisi format yang berlaku.
//
//   landcover --mode cpu|opt [--kmeans auto|omp|cuda] --input <png|folder> [--name <dataset>]
//             --k 8 --iters 20 --seed 42 --threads 8 --out <folder>
//             [--preview-max 1024] [--pixel-size 10] [--write-labels]
//   landcover --info        kemampuan build (JSON) lalu keluar
//
// Kode keluar: 0 sukses; 1 galat saat berjalan (masukan rusak, memori, K-Means gagal, ...);
// 2 argumen salah; 3 mesin K-Means yang diminta tidak terhubung pada build ini.
// Pesan galat ke stderr, baris progres ke stdout (format sama dengan MockLauncher):
//   [stage] <baca|inisialisasi|kmeans|pewarnaan|tulis> selesai <detik> s
//
// Pembagian waktu (timing.json, ms), aturan SAMA di mode cpu dan opt:
//   read_ms    dari awal membaca potongan PNG sampai buffer RGB penuh
//   init_ms, kmeans_ms, h2d/kernel/d2h_ms   dari StageTimes yang diisi kmeans_*
//   color_ms   label -> warna centroid, menimpa buffer RGB di tempat
//   write_ms   map.png (PNG RGB resolusi penuh) sampai ditulis
//   total_ms   read + waktu dinding pemanggilan kmeans_* + color + write
//   aux_ms     (kunci tambahan) counts, pratinjau, dan labels.u8; TIDAK masuk total_ms,
//              supaya --write-labels tidak mengubah angka benchmark
//
// Tautan: kmeans_cpu, kmeans_omp, kmeans_cuda, dan ref_kmeans dideklarasikan WEAK. Build
// tanpa salah satunya tetap tertaut; mesinnya dilaporkan tidak tersedia (kode keluar 3
// untuk --kmeans yang diminta eksplisit). ref_kmeans (tests/) menjadi pengganti L0 sampai
// kmeans_cpu Azka ada.
//
// Folder --out: berkas run lama (termasuk labels.u8, verification.json, map.png basi) dan
// sisa *.tmp dihapus SETELAH masukan dan mesin divalidasi, SEBELUM menulis apa pun. Tiap
// berkas ditulis ke .tmp lalu di-rename, meta.json terakhir (tanda run lengkap). Run yang
// gagal di tengah meninggalkan folder tanpa meta.json, jadi tidak pernah tampak valid.

#include "json_mini.h"
#include "kmeans.h"
#include "png_io.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" int ref_kmeans(const uint8_t* rgb, int32_t* labels, uint8_t* centroids,
                          const KMeansParams* p, StageTimes* t);

#pragma weak kmeans_cpu
#pragma weak kmeans_omp
#pragma weak kmeans_cuda
#pragma weak ref_kmeans

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kExitOk = 0;
constexpr int kExitRuntime = 1;
constexpr int kExitUsage = 2;
constexpr int kExitEngine = 3;

// Ambang SEMENTARA untuk --kmeans auto (sama dengan MockLauncher). Ambang sebenarnya dari
// Gavra (#10); ganti konstanta ini dan teks alasannya bila sudah ada.
constexpr uint64_t kAutoCudaThresholdPixels = 8000000ULL;

constexpr int kContractVersion = 1;

#ifdef _OPENMP
constexpr bool kHasOpenMP = true;
#else
constexpr bool kHasOpenMP = false;
#endif

struct UsageError : std::runtime_error { using std::runtime_error::runtime_error; };
struct EngineError : std::runtime_error { using std::runtime_error::runtime_error; };

// ---------------------------------------------------------------------------------------
// Utilitas kecil
// ---------------------------------------------------------------------------------------

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double clean_ms(double v) { return (std::isfinite(v) && v > 0.0) ? v : 0.0; }

#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
void appendf(std::string& out, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (static_cast<size_t>(n) < sizeof(buf)) {
        out.append(buf, static_cast<size_t>(n));
        return;
    }
    std::string big(static_cast<size_t>(n) + 1, '\0');
    va_start(ap, fmt);
    std::vsnprintf(&big[0], big.size(), fmt, ap);
    va_end(ap);
    out.append(big.data(), static_cast<size_t>(n));
}

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fputc('\n', stdout);
    std::fflush(stdout);  // UI membaca per baris, jangan tertahan di buffer
}

void stage_done(const char* name, double ms) {
    say("[stage] %s selesai %.2f s", name, ms / 1000.0);
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (const unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) appendf(o, "\\u%04x", static_cast<unsigned>(c));
                else o += static_cast<char>(c);
        }
    }
    o += '"';
    return o;
}

std::string json_num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.15g", std::isfinite(v) ? v : 0.0);
    return buf;
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool has_png_ext(const fs::path& p) { return lower(p.extension().string()) == ".png"; }

// ---------------------------------------------------------------------------------------
// Argumen
// ---------------------------------------------------------------------------------------

struct Options {
    bool info = false;
    bool help = false;
    std::string mode;
    std::string kmeans = "auto";
    std::vector<std::string> inputs;
    std::string name;
    int k = 8;
    int iters = 20;
    uint32_t seed = 42;
    int threads = 0;  // 0 = belum diisi
    std::string out;
    int preview_max = 1024;
    double pixel_size = -1.0;  // < 0 = dari manifest, bila tidak ada 10
    bool write_labels = false;

    bool opt() const { return mode == "opt"; }
};

const char* kUsage =
    "pemakaian:\n"
    "  landcover --mode cpu|opt [--kmeans auto|omp|cuda] --input <png|folder> [--name <dataset>]\n"
    "            --k 8 --iters 20 --seed 42 --threads 8 --out <folder>\n"
    "            [--preview-max 1024] [--pixel-size 10] [--write-labels]\n"
    "  landcover --info     tampilkan kemampuan build (JSON)\n"
    "  landcover --help\n"
    "kode keluar: 0 sukses, 1 galat saat berjalan, 2 argumen salah, 3 mesin K-Means tidak tersedia\n";

long long parse_int(const std::string& flag, const std::string& text, long long lo, long long hi) {
    if (text.empty()) throw UsageError(flag + ": nilai kosong");
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        throw UsageError(flag + ": harus bilangan bulat, ditemukan '" + text + "'");
    if (v < lo || v > hi)
        throw UsageError(flag + ": harus " + std::to_string(lo) + ".." + std::to_string(hi) +
                         ", ditemukan " + text);
    return v;
}

double parse_positive_double(const std::string& flag, const std::string& text) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(text.c_str(), &end);
    if (text.empty() || errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(v) ||
        v <= 0.0)
        throw UsageError(flag + ": harus angka > 0, ditemukan '" + text + "'");
    return v;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { o.help = true; continue; }
        if (arg.rfind("--", 0) != 0) throw UsageError("argumen tak dikenal: '" + arg + "'");
        std::string flag = arg, inline_value;
        bool has_inline = false;
        const size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            flag = arg.substr(0, eq);
            inline_value = arg.substr(eq + 1);
            has_inline = true;
        }
        auto value = [&]() -> std::string {
            if (has_inline) return inline_value;
            if (i + 1 >= argc) throw UsageError(flag + ": nilai diperlukan");
            return std::string(argv[++i]);
        };
        auto no_value = [&]() {
            if (has_inline) throw UsageError(flag + ": tidak menerima nilai");
        };
        if (flag == "--info") { no_value(); o.info = true; }
        else if (flag == "--write-labels") { no_value(); o.write_labels = true; }
        else if (flag == "--mode") o.mode = value();
        else if (flag == "--kmeans") o.kmeans = value();
        else if (flag == "--input") o.inputs.push_back(value());
        else if (flag == "--name") o.name = value();
        else if (flag == "--out") o.out = value();
        else if (flag == "--k") o.k = static_cast<int>(parse_int(flag, value(), 1, KMEANS_MAX_K));
        else if (flag == "--iters") o.iters = static_cast<int>(parse_int(flag, value(), 1, 1000000));
        else if (flag == "--seed") o.seed = static_cast<uint32_t>(parse_int(flag, value(), 0, 4294967295LL));
        else if (flag == "--threads") o.threads = static_cast<int>(parse_int(flag, value(), 1, 4096));
        else if (flag == "--preview-max") o.preview_max = static_cast<int>(parse_int(flag, value(), 1, 8192));
        else if (flag == "--pixel-size") o.pixel_size = parse_positive_double(flag, value());
        else throw UsageError("flag tak dikenal: " + flag);
    }
    if (o.help || o.info) return o;

    if (o.mode != "cpu" && o.mode != "opt")
        throw UsageError("--mode wajib: cpu atau opt (ditemukan '" + o.mode + "')");
    if (o.kmeans != "auto" && o.kmeans != "omp" && o.kmeans != "cuda")
        throw UsageError("--kmeans harus auto, omp, atau cuda (ditemukan '" + o.kmeans + "')");
    if (o.inputs.empty()) throw UsageError("--input wajib");
    if (o.inputs.size() > 1)
        throw UsageError("beberapa --input (batch beberapa wilayah) belum didukung");
    if (o.inputs[0].empty()) throw UsageError("--input kosong");
    if (o.out.empty()) throw UsageError("--out wajib");
    if (o.threads == 0) {
#ifdef _OPENMP
        o.threads = std::max(1, omp_get_max_threads());
#else
        o.threads = 1;
#endif
    }
    return o;
}

// ---------------------------------------------------------------------------------------
// Dataset: satu PNG, atau folder dengan manifest.json + potongan PNG
// ---------------------------------------------------------------------------------------

struct Chunk {
    std::string path;
    int x = 0, y = 0, w = 0, h = 0;
};

struct Dataset {
    std::string name;
    std::string source;  // teks bebas dari manifest (boleh kosong)
    int width = 0, height = 0;
    double pixel_size_m = 10.0;
    std::vector<Chunk> chunks;
};

constexpr size_t kMaxChunks = 4096;
constexpr size_t kMaxManifestBytes = 64u << 20;

std::string read_text_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error(p.string() + ": tidak bisa dibuka");
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    if (s.size() > kMaxManifestBytes) throw std::runtime_error(p.string() + ": terlalu besar");
    return s;
}

std::string leaf_name(const fs::path& p) {
    fs::path a = fs::absolute(p).lexically_normal();
    if (!a.has_filename()) a = a.parent_path();
    return a.filename().string();
}

long long manifest_int(const lc::Json& obj, const char* key, const std::string& ctx, long long lo,
                       long long hi) {
    const lc::Json* v = obj.find(key);
    if (v == nullptr) throw std::runtime_error(ctx + ": kunci '" + key + "' tidak ada");
    if (v->type != lc::Json::Type::Number || !v->is_integer)
        throw std::runtime_error(ctx + ": '" + key + "' harus bilangan bulat");
    if (v->integer < lo || v->integer > hi)
        throw std::runtime_error(ctx + ": '" + key + "' di luar rentang " + std::to_string(lo) +
                                 ".." + std::to_string(hi) + " (" + std::to_string(v->integer) + ")");
    return v->integer;
}

Dataset load_manifest(const fs::path& manifest, const fs::path& folder) {
    const std::string ctx = manifest.string();
    lc::Json root;
    std::string err;
    if (!lc::json_parse(read_text_file(manifest), root, err))
        throw std::runtime_error(ctx + ": JSON tidak sah: " + err);
    if (root.type != lc::Json::Type::Object)
        throw std::runtime_error(ctx + ": harus berupa objek JSON");

    Dataset ds;
    ds.width = static_cast<int>(manifest_int(root, "width", ctx, 1, INT_MAX));
    ds.height = static_cast<int>(manifest_int(root, "height", ctx, 1, INT_MAX));
    if (const lc::Json* ps = root.find("pixel_size_m")) {
        if (ps->type != lc::Json::Type::Number || !std::isfinite(ps->number) || ps->number <= 0.0)
            throw std::runtime_error(ctx + ": 'pixel_size_m' harus angka > 0");
        ds.pixel_size_m = ps->number;
    }
    if (const lc::Json* src = root.find("source")) {
        if (src->type == lc::Json::Type::String) ds.source = src->string;
    }

    const lc::Json* chunks = root.find("chunks");
    if (chunks == nullptr || chunks->type != lc::Json::Type::Array || chunks->array.empty())
        throw std::runtime_error(ctx + ": 'chunks' harus daftar potongan yang tidak kosong");
    if (chunks->array.size() > kMaxChunks)
        throw std::runtime_error(ctx + ": terlalu banyak potongan (maks " +
                                 std::to_string(kMaxChunks) + ")");

    const uint64_t total = static_cast<uint64_t>(ds.width) * static_cast<uint64_t>(ds.height);
    uint64_t covered = 0;
    for (size_t i = 0; i < chunks->array.size(); ++i) {
        const lc::Json& cj = chunks->array[i];
        const std::string cctx = ctx + ": chunks[" + std::to_string(i) + "]";
        if (cj.type != lc::Json::Type::Object) throw std::runtime_error(cctx + " harus objek");
        const lc::Json* file = cj.find("file");
        if (file == nullptr || file->type != lc::Json::Type::String || file->string.empty())
            throw std::runtime_error(cctx + ": 'file' harus teks tidak kosong");
        const fs::path fp(file->string);
        bool unsafe = fp.is_absolute() || fp.has_root_name();
        for (const auto& part : fp)
            if (part == "..") unsafe = true;
        if (unsafe)
            throw std::runtime_error(cctx + ": 'file' harus path relatif di dalam folder dataset");

        Chunk c;
        c.path = (folder / fp).string();
        c.x = static_cast<int>(manifest_int(cj, "x", cctx, 0, INT_MAX));
        c.y = static_cast<int>(manifest_int(cj, "y", cctx, 0, INT_MAX));
        c.w = static_cast<int>(manifest_int(cj, "w", cctx, 1, INT_MAX));
        c.h = static_cast<int>(manifest_int(cj, "h", cctx, 1, INT_MAX));
        if (static_cast<long long>(c.x) + c.w > ds.width || static_cast<long long>(c.y) + c.h > ds.height)
            throw std::runtime_error(cctx + ": potongan keluar dari citra " +
                                     std::to_string(ds.width) + "x" + std::to_string(ds.height));
        const uint64_t area = static_cast<uint64_t>(c.w) * static_cast<uint64_t>(c.h);
        if (area > total - covered)
            throw std::runtime_error(ctx + ": luas potongan melebihi luas citra (potongan saling tumpang tindih)");
        covered += area;
        ds.chunks.push_back(std::move(c));
    }
    // Semua potongan di dalam batas dan jumlah luasnya <= luas citra; tumpang tindih tetap perlu
    // diperiksa langsung (celah dan tumpang tindih bisa saling menutupi dalam jumlah luas).
    for (size_t i = 0; i < ds.chunks.size(); ++i) {
        for (size_t j = i + 1; j < ds.chunks.size(); ++j) {
            const Chunk& a = ds.chunks[i];
            const Chunk& b = ds.chunks[j];
            if (static_cast<long long>(a.x) < static_cast<long long>(b.x) + b.w &&
                static_cast<long long>(b.x) < static_cast<long long>(a.x) + a.w &&
                static_cast<long long>(a.y) < static_cast<long long>(b.y) + b.h &&
                static_cast<long long>(b.y) < static_cast<long long>(a.y) + a.h)
                throw std::runtime_error(ctx + ": chunks[" + std::to_string(i) + "] dan chunks[" +
                                         std::to_string(j) + "] saling tumpang tindih");
        }
    }
    if (covered != total)
        throw std::runtime_error(ctx + ": potongan hanya menutup " + std::to_string(covered) +
                                 " dari " + std::to_string(total) + " piksel citra (ada celah)");
    return ds;
}

Dataset load_dataset(const Options& o) {
    const fs::path in(o.inputs[0]);
    std::error_code ec;
    Dataset ds;
    std::string default_name;

    if (fs::is_directory(in, ec)) {
        default_name = leaf_name(in);
        const fs::path manifest = in / "manifest.json";
        if (fs::is_regular_file(manifest, ec)) {
            ds = load_manifest(manifest, in);
        } else {
            std::vector<fs::path> pngs;
            for (const auto& e : fs::directory_iterator(in, ec))
                if (e.is_regular_file(ec) && has_png_ext(e.path())) pngs.push_back(e.path());
            if (pngs.size() != 1)
                throw std::runtime_error(in.string() + ": tanpa manifest.json harus berisi tepat satu PNG (ditemukan " +
                                         std::to_string(pngs.size()) + ")");
            Chunk c;
            c.path = pngs[0].string();
            ds.chunks.push_back(c);
        }
    } else if (fs::is_regular_file(in, ec)) {
        if (!has_png_ext(in)) throw std::runtime_error(in.string() + ": bukan berkas .png");
        default_name = in.stem().string();
        Chunk c;
        c.path = in.string();
        ds.chunks.push_back(c);
    } else {
        throw std::runtime_error(in.string() + ": masukan tidak ditemukan");
    }

    // Header setiap potongan diperiksa sekarang (murah), sebelum folder keluaran disentuh.
    for (size_t i = 0; i < ds.chunks.size(); ++i) {
        Chunk& c = ds.chunks[i];
        int w = 0, h = 0;
        std::string err;
        if (!lc::png_info(c.path, w, h, err)) throw std::runtime_error(err);
        if (ds.width == 0) {  // satu PNG tanpa manifest: ukuran dari header
            ds.width = w;
            ds.height = h;
            c.w = w;
            c.h = h;
        } else if (w != c.w || h != c.h) {
            throw std::runtime_error(c.path + ": ukuran PNG " + std::to_string(w) + "x" +
                                     std::to_string(h) + " tidak sama dengan manifest (" +
                                     std::to_string(c.w) + "x" + std::to_string(c.h) + ")");
        }
    }
    ds.name = o.name.empty() ? default_name : o.name;
    if (ds.name.empty()) throw std::runtime_error("nama dataset kosong; beri --name");
    if (o.pixel_size > 0.0) ds.pixel_size_m = o.pixel_size;
    return ds;
}

// ---------------------------------------------------------------------------------------
// Mesin K-Means
// ---------------------------------------------------------------------------------------

using KmeansFn = int (*)(const uint8_t*, int32_t*, uint8_t*, const KMeansParams*, StageTimes*);

struct Engine {
    KmeansFn fn = nullptr;
    std::string device;  // "serial" | "omp" | "cuda" (nilai meta.kmeans_device)
    std::string impl;    // nama fungsi yang benar-benar dipanggil
    std::string reason;  // kmeans_auto_reason (hanya --kmeans auto)
    std::string note;    // catatan pengganti/cadangan
};

std::string juta_text(uint64_t n) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(n) / 1e6);
    return buf;
}

Engine serial_engine() {
    Engine e;
    e.device = "serial";
    if (KmeansFn f = kmeans_cpu) {
        e.fn = f;
        e.impl = "kmeans_cpu";
    } else if (KmeansFn r = ref_kmeans) {
        e.fn = r;
        e.impl = "ref_kmeans";
        e.note = "kmeans_cpu (L0 Azka) belum terhubung pada build ini; memakai tests/ref_kmeans.cpp sebagai pengganti";
    } else {
        throw EngineError("tidak ada implementasi K-Means CPU yang terhubung (kmeans_cpu atau ref_kmeans)");
    }
    return e;
}

Engine omp_engine() {
    Engine e;
    e.fn = kmeans_omp;
    e.device = "omp";
    e.impl = "kmeans_omp";
    return e;
}

Engine cuda_engine() {
    Engine e;
    e.fn = kmeans_cuda;
    e.device = "cuda";
    e.impl = "kmeans_cuda";
    return e;
}

Engine select_engine(const Options& o, uint64_t n) {
    if (!o.opt()) return serial_engine();
    const bool have_omp = kmeans_omp != nullptr;
    const bool have_cuda = kmeans_cuda != nullptr;
    if (o.kmeans == "cuda") {
        if (!have_cuda)
            throw EngineError("--kmeans cuda: kmeans_cuda tidak terhubung pada build ini (butuh nvcc dan src/cuda/*.cu saat make landcover)");
        return cuda_engine();
    }
    if (o.kmeans == "omp") {
        if (!have_omp)
            throw EngineError("--kmeans omp: kmeans_omp tidak terhubung pada build ini (src/omp/*.cpp belum ada saat make landcover)");
        return omp_engine();
    }
    // auto
    const std::string basis = "N = " + juta_text(n) + " juta piksel";
    const std::string limit = std::to_string(kAutoCudaThresholdPixels / 1000000ULL) + " juta (ambang sementara, belum dari Gavra #10)";
    Engine e;
    if (n >= kAutoCudaThresholdPixels) {
        if (have_cuda) {
            e = cuda_engine();
            e.reason = basis + " >= ambang " + limit;
            return e;
        }
        if (have_omp) {
            e = omp_engine();
            e.reason = basis + " >= ambang " + limit + ", tetapi CUDA tidak terhubung pada build ini; memakai omp";
            return e;
        }
    } else if (have_omp) {
        e = omp_engine();
        e.reason = basis + " < ambang " + limit;
        return e;
    }
    e = serial_engine();
    e.reason = basis + (n >= kAutoCudaThresholdPixels ? " >= " : " < ") + "ambang " + limit +
               ", tetapi kmeans_omp dan kmeans_cuda tidak terhubung pada build ini; jatuh ke serial";
    return e;
}

// ---------------------------------------------------------------------------------------
// Berkas keluaran
// ---------------------------------------------------------------------------------------

const char* const kRunFiles[] = {
    "meta.json",         "timing.json",  "centroids.json", "counts.json", "labels_preview.png",
    "map_preview.png",   "map.png",      "labels.u8",      "verification.json",
};

void remove_tmp_files(const fs::path& dir) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file(ec) && e.path().extension() == ".tmp") fs::remove(e.path(), ec);
    }
}

void prepare_out_dir(const fs::path& out, const Dataset& ds) {
    std::error_code ec;
    if (fs::exists(out, ec) && !fs::is_directory(out, ec))
        throw std::runtime_error(out.string() + ": ada tetapi bukan folder");
    // Pembersihan menghapus nama-nama berkas run; jangan sampai itu menghapus MASUKAN
    // (mis. --out = folder dataset yang berisi potongan bernama map.png).
    for (const Chunk& c : ds.chunks) {
        const fs::path cp(c.path);
        const std::string leaf = lower(cp.filename().string());
        for (const char* name : kRunFiles) {
            if (leaf == name && fs::equivalent(cp, out / name, ec))
                throw std::runtime_error(c.path + ": masukan akan tertimpa oleh keluaran; pilih --out yang lain");
        }
    }
    fs::create_directories(out, ec);
    if (ec) throw std::runtime_error(out.string() + ": gagal membuat folder (" + ec.message() + ")");
    for (const char* name : kRunFiles) {
        const fs::path p = out / name;
        if (fs::is_regular_file(p, ec)) {
            fs::remove(p, ec);
            if (ec) throw std::runtime_error(p.string() + ": gagal menghapus berkas run lama (" + ec.message() + ")");
        } else if (fs::exists(p, ec)) {
            throw std::runtime_error(p.string() + ": ada tetapi bukan berkas biasa");
        }
    }
    remove_tmp_files(out);
}

void commit_tmp(const fs::path& tmp, const fs::path& final_path) {
    std::error_code ec;
    fs::rename(tmp, final_path, ec);
    if (ec) throw std::runtime_error(final_path.string() + ": gagal mengganti nama dari .tmp (" + ec.message() + ")");
}

fs::path tmp_of(const fs::path& p) { return fs::path(p.string() + ".tmp"); }

void write_text_atomic(const fs::path& path, const std::string& content) {
    const fs::path tmp = tmp_of(path);
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (f == nullptr) throw std::runtime_error(path.string() + ": tidak bisa ditulis");
    const bool ok = std::fwrite(content.data(), 1, content.size(), f) == content.size();
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) throw std::runtime_error(path.string() + ": gagal menulis");
    commit_tmp(tmp, path);
}

void write_png_atomic(const fs::path& path, const uint8_t* px, int w, int h, bool rgb) {
    const fs::path tmp = tmp_of(path);
    std::string err;
    const bool ok = rgb ? lc::png_write_rgb8(tmp.string(), px, w, h, err)
                        : lc::png_write_gray8(tmp.string(), px, w, h, err);
    if (!ok) throw std::runtime_error(err);
    commit_tmp(tmp, path);
}

void write_labels_u8(const fs::path& path, const int32_t* labels, size_t n) {
    const fs::path tmp = tmp_of(path);
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (f == nullptr) throw std::runtime_error(path.string() + ": tidak bisa ditulis");
    std::vector<uint8_t> buf(size_t{1} << 20);
    bool ok = true;
    for (size_t off = 0; off < n && ok; off += buf.size()) {
        const size_t m = std::min(buf.size(), n - off);
        for (size_t i = 0; i < m; ++i) buf[i] = static_cast<uint8_t>(labels[off + i]);
        ok = std::fwrite(buf.data(), 1, m, f) == m;
    }
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) throw std::runtime_error(path.string() + ": gagal menulis labels.u8");
    commit_tmp(tmp, path);
}

std::string iso_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm);
    std::string s = buf;
    if (s.size() >= 5 && (s[s.size() - 5] == '+' || s[s.size() - 5] == '-')) s.insert(s.size() - 2, ":");
    return s;
}

std::string describe_host() {
    if (const char* e = std::getenv("LC_HOST")) {
        if (*e) return e;
    }
    std::string cpu, os;
    bool wsl = false;
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("model name", 0) == 0) {
                const size_t c = line.find(':');
                if (c != std::string::npos) {
                    cpu = line.substr(c + 1);
                    cpu.erase(0, cpu.find_first_not_of(" \t"));
                }
                break;
            }
        }
    }
    {
        std::ifstream f("/proc/version");
        std::string line;
        if (std::getline(f, line)) wsl = lower(line).find("microsoft") != std::string::npos;
    }
    {
        std::ifstream f("/etc/os-release");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("PRETTY_NAME=", 0) == 0) {
                os = line.substr(12);
                if (os.size() >= 2 && os.front() == '"' && os.back() == '"') os = os.substr(1, os.size() - 2);
                break;
            }
        }
    }
    std::string host = cpu;
    const std::string tail = std::string(wsl ? "WSL2 " : "") + os;
    if (!tail.empty()) host += (host.empty() ? "" : ", ") + tail;
    return host.empty() ? "tidak diketahui" : host;
}

// ---------------------------------------------------------------------------------------
// Kernel pasca-K-Means
// ---------------------------------------------------------------------------------------

bool count_labels(const int32_t* labels, size_t n, int k, bool parallel, uint64_t* counts) {
    for (int c = 0; c < KMEANS_MAX_K; ++c) counts[c] = 0;
    bool ok = true;
    const ptrdiff_t len = static_cast<ptrdiff_t>(n);
#pragma omp parallel if (parallel)
    {
        uint64_t local[KMEANS_MAX_K] = {0};
        bool bad = false;
#pragma omp for schedule(static) nowait
        for (ptrdiff_t i = 0; i < len; ++i) {
            const int32_t l = labels[i];
            if (static_cast<uint32_t>(l) >= static_cast<uint32_t>(k)) bad = true;
            else ++local[l];
        }
#pragma omp critical(lc_counts)
        {
            for (int c = 0; c < KMEANS_MAX_K; ++c) counts[c] += local[c];
            if (bad) ok = false;
        }
    }
    return ok;
}

void colorize(uint8_t* rgb, const int32_t* labels, size_t n, const uint8_t* centroids, bool parallel) {
    uint8_t pal[KMEANS_MAX_K][3];
    std::memcpy(pal, centroids, sizeof(pal));
    const ptrdiff_t len = static_cast<ptrdiff_t>(n);
#pragma omp parallel for schedule(static) if (parallel)
    for (ptrdiff_t i = 0; i < len; ++i) {
        const uint8_t* c = pal[labels[i]];
        uint8_t* d = rgb + static_cast<size_t>(i) * 3;
        d[0] = c[0];
        d[1] = c[1];
        d[2] = c[2];
    }
}

struct PreviewDims { int w, h; };

PreviewDims preview_dims(int w, int h, int pmax) {
    const int64_t m = std::max(w, h);
    if (m <= pmax) return {w, h};
    auto scaled = [&](int v) {
        return static_cast<int>(std::max<int64_t>(1, (static_cast<int64_t>(v) * pmax + m / 2) / m));
    };
    return {scaled(w), scaled(h)};
}

// Sampling nearest: piksel pratinjau (x, y) mengambil piksel sumber di pusat selnya.
void build_previews(const int32_t* labels, int w, int h, const uint8_t* centroids, PreviewDims pd,
                    std::vector<uint8_t>& lab, std::vector<uint8_t>& map) {
    lab.assign(static_cast<size_t>(pd.w) * pd.h, 0);
    map.assign(static_cast<size_t>(pd.w) * pd.h * 3, 0);
    std::vector<size_t> sx(static_cast<size_t>(pd.w));
    for (int x = 0; x < pd.w; ++x)
        sx[x] = static_cast<size_t>((static_cast<uint64_t>(2 * x + 1) * static_cast<uint64_t>(w)) /
                                    (2ull * static_cast<uint64_t>(pd.w)));
    for (int y = 0; y < pd.h; ++y) {
        const size_t sy = static_cast<size_t>((static_cast<uint64_t>(2 * y + 1) * static_cast<uint64_t>(h)) /
                                              (2ull * static_cast<uint64_t>(pd.h)));
        const int32_t* row = labels + sy * static_cast<size_t>(w);
        uint8_t* lrow = &lab[static_cast<size_t>(y) * pd.w];
        uint8_t* mrow = &map[static_cast<size_t>(y) * pd.w * 3];
        for (int x = 0; x < pd.w; ++x) {
            const int32_t l = row[sx[x]];
            lrow[x] = static_cast<uint8_t>(l);
            std::memcpy(mrow + static_cast<size_t>(x) * 3, centroids + static_cast<size_t>(l) * 3, 3);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Membaca citra ke buffer RGB penuh
// ---------------------------------------------------------------------------------------

using RgbBuffer = std::unique_ptr<uint8_t, void (*)(void*)>;

// Mengembalikan true bila potongan dibaca PARALEL (untuk dilaporkan sebagai perangkat "omp").
bool read_image(const Dataset& ds, bool allow_parallel, RgbBuffer& rgb) {
    const size_t W = static_cast<size_t>(ds.width), H = static_cast<size_t>(ds.height);
    if (ds.chunks.size() == 1) {
        const Chunk& c = ds.chunks[0];
        if (c.x == 0 && c.y == 0 && c.w == ds.width && c.h == ds.height) {
            lc::Rgb8 img;
            std::string err;
            if (!lc::png_read_rgb8(c.path, img, err)) throw std::runtime_error(err);
            if (img.width != ds.width || img.height != ds.height)
                throw std::runtime_error(c.path + ": ukuran setelah dekode tidak sama dengan header");
            rgb = RgbBuffer(img.data.release(), std::free);
            return false;
        }
    }
    uint8_t* raw = static_cast<uint8_t*>(std::malloc(W * H * 3));
    if (raw == nullptr) throw std::bad_alloc();
    rgb = RgbBuffer(raw, std::free);

    const bool par = allow_parallel && ds.chunks.size() > 1;
    std::atomic<bool> failed{false};
    std::mutex err_mu;
    std::string first_err;
    const int nchunks = static_cast<int>(ds.chunks.size());
#pragma omp parallel for schedule(dynamic, 1) if (par)
    for (int ci = 0; ci < nchunks; ++ci) {
        if (failed.load()) continue;
        const Chunk& c = ds.chunks[static_cast<size_t>(ci)];
        lc::Rgb8 img;
        std::string err;
        bool ok = lc::png_read_rgb8(c.path, img, err);
        if (ok && (img.width != c.w || img.height != c.h)) {
            ok = false;
            err = c.path + ": ukuran setelah dekode tidak sama dengan manifest";
        }
        if (!ok) {
            std::lock_guard<std::mutex> lock(err_mu);
            if (!failed.exchange(true)) first_err = err;
            continue;
        }
        const uint8_t* src = img.data.get();
        for (int row = 0; row < c.h; ++row) {
            std::memcpy(raw + ((static_cast<size_t>(c.y) + static_cast<size_t>(row)) * W + static_cast<size_t>(c.x)) * 3,
                        src + static_cast<size_t>(row) * static_cast<size_t>(c.w) * 3,
                        static_cast<size_t>(c.w) * 3);
        }
    }
    if (failed.load()) throw std::runtime_error(first_err);
    return par && kHasOpenMP;
}

// ---------------------------------------------------------------------------------------
// --info
// ---------------------------------------------------------------------------------------

void print_info() {
    const char* cpu = kmeans_cpu != nullptr ? "kmeans_cpu" : (ref_kmeans != nullptr ? "ref_kmeans" : nullptr);
    std::string s = "{\n";
    appendf(s, "  \"program\": \"landcover\",\n  \"contract_version\": %d,\n", kContractVersion);
    appendf(s, "  \"openmp\": %s,\n", kHasOpenMP ? "true" : "false");
    int maxt = 1;
#ifdef _OPENMP
    maxt = omp_get_max_threads();
#endif
    appendf(s, "  \"max_threads\": %d,\n", maxt);
    s += "  \"engines\": {\n";
    appendf(s, "    \"cpu\": %s,\n", cpu ? json_str(cpu).c_str() : "null");
    appendf(s, "    \"omp\": %s,\n", kmeans_omp != nullptr ? "true" : "false");
    appendf(s, "    \"cuda\": %s\n  },\n", kmeans_cuda != nullptr ? "true" : "false");
    appendf(s, "  \"auto_cuda_threshold_pixels\": %llu\n}\n", static_cast<unsigned long long>(kAutoCudaThresholdPixels));
    std::fputs(s.c_str(), stdout);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------

int run(const Options& o) {
    const bool opt = o.opt();
    const int threads = opt ? o.threads : 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    // 1. Validasi: masukan dan mesin, sebelum folder keluaran disentuh.
    const Dataset ds = load_dataset(o);
    const uint64_t n64 = static_cast<uint64_t>(ds.width) * static_cast<uint64_t>(ds.height);
    if (n64 > static_cast<uint64_t>(SIZE_MAX / 4))
        throw std::runtime_error("citra terlalu besar untuk memori proses ini");
    const size_t n = static_cast<size_t>(n64);
    Engine engine = select_engine(o, n64);

    say("[info] dataset '%s': %dx%d piksel, %zu potongan, %g m/piksel", ds.name.c_str(), ds.width,
        ds.height, ds.chunks.size(), ds.pixel_size_m);
    say("[info] mesin K-Means: %s (%s), mode %s, K=%d, iterasi=%d, seed=%u, thread=%d",
        engine.impl.c_str(), engine.device.c_str(), o.mode.c_str(), o.k, o.iters,
        static_cast<unsigned>(o.seed), threads);
    if (!engine.reason.empty()) say("[info] auto: %s", engine.reason.c_str());
    if (!engine.note.empty()) say("[peringatan] %s", engine.note.c_str());

    const fs::path out(o.out);
    prepare_out_dir(out, ds);
    try {
        // 2. Baca.
        RgbBuffer rgb(nullptr, std::free);
        auto t0 = Clock::now();
        const bool read_parallel = read_image(ds, opt && kHasOpenMP && threads > 1, rgb);
        const double read_ms = ms_since(t0);
        stage_done("baca", read_ms);

        // 3. Centroid awal untuk meta (kmeans_* menghitung ulang sendiri; K pilihan, murah).
        uint8_t init_c[KMEANS_MAX_K * 3] = {0};
        if (kmeans_init_centroids(rgb.get(), n, o.k, o.seed, init_c) != 0)
            throw std::runtime_error("kmeans_init_centroids gagal (argumen tidak sah)");

        // 4. K-Means.
        std::unique_ptr<int32_t[]> labels(new int32_t[n]);
        uint8_t centroids[KMEANS_MAX_K * 3] = {0};
        StageTimes st;
        std::memset(&st, 0, sizeof(st));
        KMeansParams kp;
        kp.width = ds.width;
        kp.height = ds.height;
        kp.k = o.k;
        kp.max_iter = o.iters;
        kp.seed = o.seed;
        kp.num_threads = (engine.device == "omp") ? threads : 1;

        t0 = Clock::now();
        int rc = engine.fn(rgb.get(), labels.get(), centroids, &kp, &st);
        if (rc != 0 && opt && o.kmeans == "auto" && engine.device == "cuda") {
            const std::string why = "kmeans_cuda gagal (kode " + std::to_string(rc) + ")";
            Engine fb = (kmeans_omp != nullptr) ? omp_engine() : serial_engine();
            fb.reason = engine.reason + "; " + why + ", memakai " + fb.device;
            engine = fb;
            say("[peringatan] %s; mengulang dengan %s", why.c_str(), engine.impl.c_str());
            kp.num_threads = (engine.device == "omp") ? threads : 1;
            std::memset(&st, 0, sizeof(st));
            rc = engine.fn(rgb.get(), labels.get(), centroids, &kp, &st);
        }
        const double kcall_ms = ms_since(t0);
        if (rc != 0)
            throw std::runtime_error("K-Means (" + engine.impl + ") gagal dengan kode " + std::to_string(rc));
        const bool cuda = engine.device == "cuda";
        const double init_ms = clean_ms(st.init_ms);
        const double kmeans_ms = clean_ms(st.kmeans_ms);
        stage_done("inisialisasi", init_ms);
        stage_done("kmeans", kmeans_ms);

        // 5. Pekerjaan pendamping UI (tidak masuk total_ms): counts, pratinjau, labels.u8.
        const bool par = opt && kHasOpenMP && threads > 1;
        t0 = Clock::now();
        uint64_t counts[KMEANS_MAX_K];
        if (!count_labels(labels.get(), n, o.k, par, counts))
            throw std::runtime_error("K-Means mengembalikan label di luar rentang 0.." + std::to_string(o.k - 1));
        uint64_t sum = 0;
        for (int c = 0; c < o.k; ++c) sum += counts[c];
        if (sum != n64) throw std::runtime_error("jumlah counts tidak sama dengan jumlah piksel (galat internal)");

        const PreviewDims pd = preview_dims(ds.width, ds.height, o.preview_max);
        std::vector<uint8_t> prev_labels, prev_map;
        build_previews(labels.get(), ds.width, ds.height, centroids, pd, prev_labels, prev_map);
        write_png_atomic(out / "labels_preview.png", prev_labels.data(), pd.w, pd.h, false);
        write_png_atomic(out / "map_preview.png", prev_map.data(), pd.w, pd.h, true);
        if (o.write_labels) write_labels_u8(out / "labels.u8", labels.get(), n);
        const double aux_ms = ms_since(t0);

        // 6. Pewarnaan: label -> warna centroid, menimpa buffer RGB (citra asli tidak
        //    dibutuhkan lagi).
        t0 = Clock::now();
        colorize(rgb.get(), labels.get(), n, centroids, par);
        const double color_ms = ms_since(t0);
        stage_done("pewarnaan", color_ms);

        // 7. Tulis peta resolusi penuh.
        t0 = Clock::now();
        write_png_atomic(out / "map.png", rgb.get(), ds.width, ds.height, true);
        const double write_ms = ms_since(t0);
        stage_done("tulis", write_ms);

        const double total_ms = std::max(1e-6, read_ms + kcall_ms + color_ms + write_ms);

        // 8. JSON. meta.json terakhir.
        const char* read_dev = read_parallel ? "omp" : "serial";
        const char* color_dev = par ? "omp" : "serial";
        std::string timing = "{\n  \"unit\": \"ms\",\n";
        appendf(timing, "  \"read_ms\": %.3f,\n  \"init_ms\": %.3f,\n  \"kmeans_ms\": %.3f,\n", read_ms, init_ms, kmeans_ms);
        appendf(timing, "  \"color_ms\": %.3f,\n  \"write_ms\": %.3f,\n  \"total_ms\": %.3f,\n", color_ms, write_ms, total_ms);
        appendf(timing, "  \"h2d_ms\": %.3f,\n  \"kernel_ms\": %.3f,\n  \"d2h_ms\": %.3f,\n",
                cuda ? clean_ms(st.h2d_ms) : 0.0, cuda ? clean_ms(st.kernel_ms) : 0.0,
                cuda ? clean_ms(st.d2h_ms) : 0.0);
        appendf(timing, "  \"aux_ms\": %.3f,\n", aux_ms);
        appendf(timing, "  \"devices\": {\"read\": \"%s\", \"init\": \"serial\", \"kmeans\": \"%s\", \"color\": \"%s\", \"write\": \"serial\"}\n}\n",
                read_dev, engine.device.c_str(), color_dev);
        write_text_atomic(out / "timing.json", timing);

        std::string cj = "[";
        for (int c = 0; c < o.k; ++c)
            appendf(cj, "%s[%d, %d, %d]", c ? ", " : "", centroids[c * 3], centroids[c * 3 + 1], centroids[c * 3 + 2]);
        cj += "]\n";
        write_text_atomic(out / "centroids.json", cj);

        std::string cn = "[";
        for (int c = 0; c < o.k; ++c)
            appendf(cn, "%s%llu", c ? ", " : "", static_cast<unsigned long long>(counts[c]));
        cn += "]\n";
        write_text_atomic(out / "counts.json", cn);

        std::string ic = "[";
        for (int c = 0; c < o.k; ++c)
            appendf(ic, "%s[%d, %d, %d]", c ? ", " : "", init_c[c * 3], init_c[c * 3 + 1], init_c[c * 3 + 2]);
        ic += "]";

        const int meta_threads = engine.device == "serial" ? 1 : (engine.device == "omp" ? threads : 0);
        std::string meta = "{\n";
        appendf(meta, "  \"contract_version\": %d,\n  \"mock\": false,\n", kContractVersion);
        appendf(meta, "  \"dataset\": %s,\n  \"mode\": \"%s\",\n  \"kmeans_device\": \"%s\",\n",
                json_str(ds.name).c_str(), o.mode.c_str(), engine.device.c_str());
        appendf(meta, "  \"kmeans_impl\": %s,\n", json_str(engine.impl).c_str());
        if (!engine.reason.empty()) appendf(meta, "  \"kmeans_auto_reason\": %s,\n", json_str(engine.reason).c_str());
        if (!engine.note.empty()) appendf(meta, "  \"kmeans_note\": %s,\n", json_str(engine.note).c_str());
        appendf(meta, "  \"width\": %d,\n  \"height\": %d,\n  \"pixel_size_m\": %s,\n", ds.width, ds.height,
                json_num(ds.pixel_size_m).c_str());
        appendf(meta, "  \"k\": %d,\n  \"iters\": %d,\n  \"seed\": %u,\n  \"threads\": %d,\n  \"io_threads\": %d,\n",
                o.k, o.iters, static_cast<unsigned>(o.seed), meta_threads, par ? threads : 1);
        appendf(meta, "  \"init_centroids\": %s,\n", ic.c_str());
        appendf(meta, "  \"preview_width\": %d,\n  \"preview_height\": %d,\n", pd.w, pd.h);
        appendf(meta, "  \"chunks\": %zu,\n  \"labels_written\": %s,\n", ds.chunks.size(), o.write_labels ? "true" : "false");
        if (!ds.source.empty()) appendf(meta, "  \"source\": %s,\n", json_str(ds.source).c_str());
        appendf(meta, "  \"host\": %s,\n  \"created_at\": %s\n}\n", json_str(describe_host()).c_str(),
                json_str(iso_now()).c_str());
        write_text_atomic(out / "meta.json", meta);

        say("[info] counts, pratinjau%s: %.2f s (tidak dihitung dalam total)", o.write_labels ? ", labels.u8" : "", aux_ms / 1000.0);
        say("[info] total %.2f s", total_ms / 1000.0);
        say("[info] folder run: %s", out.string().c_str());
    } catch (...) {
        remove_tmp_files(out);
        throw;
    }
    return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        if (o.help) {
            std::fputs(kUsage, stdout);
            return kExitOk;
        }
        if (o.info) {
            print_info();
            return kExitOk;
        }
        return run(o);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "landcover: argumen salah: %s\n%s", e.what(), kUsage);
        return kExitUsage;
    } catch (const EngineError& e) {
        std::fprintf(stderr, "landcover: mesin tidak tersedia: %s\n", e.what());
        return kExitEngine;
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "landcover: galat: memori tidak cukup\n");
        return kExitRuntime;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "landcover: galat: %s\n", e.what());
        return kExitRuntime;
    }
}
