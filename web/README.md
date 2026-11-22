# UI web (Vite + React + TypeScript)

Frontend aplikasi Pemetaan Tutupan Lahan. Backend: FastAPI di `app/api/` (rute `/api/*`).
Fase 1 dan 2 selesai. Halaman **Kinerja**: Bandingkan CPU saja vs Optimalisasi, kartu speedup, grafik waktu per
tahap, status verifikasi. Halaman **Pemetaan**: kartu citra (bergambar) dan unggah PNG, parameter K, "Buat peta"
(mode Optimalisasi), citra asli | peta kelas (slider sebelum/sesudah atau berdampingan), legenda yang sekaligus
editor nama kelas (Kelas dan Nama lain; Nama lain mengalahkan Kelas), tabel luas (persen dan hektare), dan menu
ekspor (PNG pratinjau, CSV luas, CSV per cluster, PNG resolusi penuh bila `labels.u8` ada). Nama kelas per run
disimpan di browser (localStorage) dan dikirim ke API; UI tidak pernah memuat citra atau label resolusi penuh.

Stack: Vite, React 19, TypeScript, Tailwind CSS v4, shadcn/ui (Radix, gaya `radix-vega`, ikon Lucide),
Recharts, TanStack Query. Teks antarmuka bahasa Indonesia; angka dan kalimat vonis datang dari server
(`app/services/`), frontend hanya menampilkan.

## Menjalankan (dari root repo, Windows)

Mode pengembangan (dua terminal):

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir app --port 8000
npm --prefix web run dev
```

Buka http://localhost:5173 (Vite meneruskan `/api` ke port 8000).

Demo satu proses (FastAPI menyajikan `web/dist`):

```powershell
npm --prefix web run build
.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir app --port 8000
```

Buka http://localhost:8000. Backend dipilih lewat `LC_BACKEND=mock|cli` (bawaan `mock`, hasil bertanda
DATA TIRUAN); `LC_MOCK_SLEEP=<detik>` memperlambat run tiruan supaya progres terlihat; backend `cli` butuh
`LC_CORE_BIN` (program `landcover`, atau pembungkus WSL).

Pemeriksaan tipe: `npm --prefix web run build` (menjalankan `tsc -b` lalu `vite build`).
