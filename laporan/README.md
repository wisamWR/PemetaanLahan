# Laporan akhir (LaTeX)

Gaya diambil dari template proposal (preamble dan sampul); isi tiap bab ada di `bab/`.

## Kompilasi

Dari folder `laporan/`:

```powershell
latexmk -pdf main.tex
```

Tanpa latexmk (butuh Perl), jalankan `pdflatex main.tex` tiga kali. Hasilnya `main.pdf`.

Semua paket yang dipakai sudah ada di distribusi TeX standar (diuji dengan MiKTeX). Jangan menambah
`\usepackage`: MiKTeX dengan pemasangan otomatis bisa mengunduh paket tanpa bertanya.

## Struktur

| Berkas | Isi |
|---|---|
| `main.tex` | Kerangka dokumen: sampul, daftar isi/gambar/tabel, bab, pustaka, lampiran |
| `preamble.tex` | Gaya dari template proposal (jangan diubah) dan tambahan laporan di bagian bawah |
| `sampul.tex` | Halaman sampul |
| `bab/bab1.tex` ... `bab/bab6.tex` | Satu berkas per bab |
| `bab/pustaka.tex` | Daftar pustaka [1]–[11] |
| `bab/lampiran.tex` | Lampiran (kontribusi, cara menjalankan, tautan repositori) |
| `gambar/` | Logo dan gambar |
| `tabel/` | Tabel hasil yang dibuat skrip (jangan disunting tangan) |
| `skrip/` | Skrip pembuat tabel |

## Tabel hasil (Bab V)

`tabel/*.tex` dan `tabel/angka.tex` (makro angka untuk teks) dibuat ulang dari akar repo dengan:

```powershell
.venv\Scripts\python.exe laporan/skrip/tabel_ke_tex.py
```

Sumbernya `bench/results/*_summary.csv`, `docs/PROFIL_L0_tabel.md`, dan `hasil_s2/` (run Sentinel-2).
Skrip berhenti bila cek silang terhadap dokumen tim gagal. Jangan menyunting `tabel/` dengan tangan.

## Aturan penulisan

- Bagian yang belum diisi ditandai `\isi{...}` (teks merah). Sebelum PDF final, pencarian `\isi{` harus kosong.
- Angka hasil hanya dari berkas data di repo, bukan diketik dari ingatan; data sintetis selalu ditandai sintetis.
- Nama berkas dan fungsi memakai `\kode{...}`; istilah asing dimiringkan dengan `\emph{...}`.
- `main.pdf` final ikut di-commit. Setelah sumber diubah, kompilasi ulang dan commit PDF-nya bersama sumber.
