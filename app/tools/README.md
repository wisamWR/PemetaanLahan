# app/tools: alat bantu menjalankan aplikasi

Dua alat untuk Windows. Keduanya hanya memanggil program lain; tidak ada logika aplikasi di sini.

| Berkas | Fungsi |
|---|---|
| `run_web.py`, `run_web.cmd` | Satu perintah untuk UI web: bangun frontend bila perlu, lalu sajikan API dan `web/dist` di satu port |
| `lc_wsl.py`, `lc_wsl.cmd` | Pembungkus: menjalankan program `landcover` yang ada di WSL dari proses Windows |

## UI web dalam satu perintah

Dari root repo:

```powershell
app\tools\run_web.cmd                       # backend tiruan, hasil bertanda DATA TIRUAN
app\tools\run_web.cmd --backend cli --open  # program landcover asli lewat WSL, buka browser
```

Sama saja: `.venv\Scripts\python.exe app\tools\run_web.py ...`. Buka http://127.0.0.1:8000.

| Opsi | Arti |
|---|---|
| `--backend mock\|cli` | `mock` (bawaan, `LC_BACKEND`) atau `cli` (program `landcover`) |
| `--data-dir`, `--runs-dir` | Folder dataset dan hasil run (`LC_DATA_DIR`, `LC_RUNS_DIR`; bawaan `data/` dan `runs/` di repo) |
| `--core-bin` | Program `landcover` atau pembungkus (`LC_CORE_BIN`). Bawaan untuk `--backend cli` di Windows: `lc_wsl.cmd` |
| `--port`, `--host` | Bawaan 8000 dan 127.0.0.1 |
| `--no-build`, `--rebuild` | Lewati atau paksa `npm run build` |

Frontend dibangun otomatis bila `web/dist` belum ada atau lebih tua daripada sumbernya. Paket npm tidak diunduh
oleh skrip ini: jalankan sekali `npm --prefix web install` (Node.js ada di Windows). Mode pengembangan frontend
(hot reload) ada di `web/README.md`.

## Menjalankan landcover asli lewat WSL

Persiapan sekali (di WSL, Ubuntu 24.04; CUDA Toolkit 12.9 bila ingin mode CUDA):

```bash
git clone "/mnt/d/<folder repo>" ~/PemetaanLahan   # atau git clone dari GitHub
cd ~/PemetaanLahan && make landcover               # build/landcover; ikut tertaut CUDA bila nvcc dan src/cuda ada
./build/landcover --info                           # periksa mesin yang terhubung
```

`lc_wsl.cmd` mengubah argumen berpath Windows (`C:\...`) menjadi path WSL (`/mnt/c/...`) lalu menjalankan biner di
dalam WSL; keluaran mengalir langsung dan kode keluar diteruskan. Pengaturan lewat environment variable:

| Variabel | Arti | Bawaan |
|---|---|---|
| `LC_WSL_BIN` | Path biner di DALAM WSL | `$HOME/PemetaanLahan/build/landcover` |
| `LC_WSL_DISTRO` | Nama distro WSL | distro bawaan |
| `LC_WSL_DRYRUN=1` | Cetak perintah `wsl.exe`, tidak menjalankan | mati |

Periksa tanpa menjalankan apa pun: `set LC_WSL_DRYRUN=1` lalu `app\tools\lc_wsl.cmd --info`.

Catatan:

- **Waktu baca/tulis.** Data dan folder run di disk Windows (`/mnt/*`) jauh lebih lambat daripada di filesystem Linux
  WSL (baca tile 10980² ±7 s vs ±3 s di uji kami). Untuk pengukuran yang akan dilaporkan, taruh `--data-dir` dan
  `--runs-dir` di `\\wsl$\Ubuntu-24.04\home\<nama>\...` atau jalankan dari dalam WSL.
- **Batas waktu.** Mematikan pembungkus (mis. karena batas waktu `CliLauncher`, `LC_CORE_TIMEOUT`) tidak mematikan
  proses `landcover` di dalam WSL; proses itu selesai sendiri.
- **Mode Optimalisasi dan `auto`.** `--kmeans auto` memilih CUDA bila N × K (piksel × cluster) ≥ 70 juta
  (misalnya ≥ 8,75 juta piksel pada K=8; dari pengukuran #12, `docs/AMBANG_AUTO.md`) dan CUDA terhubung; selain itu omp, lalu serial. Pilih K-Means "CUDA" di halaman Kinerja untuk memaksa.
