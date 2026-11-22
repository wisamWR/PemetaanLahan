# third_party/stb

Dua berkas header-only dari https://github.com/nothings/stb, tidak diubah, dipakai oleh
`src/cli/png_io_stb.cpp` (baca/tulis PNG untuk program `landcover`).

| Berkas | Versi | Ukuran (byte) | SHA-256 |
|---|---|---|---|
| `stb_image.h` | v2.34 | 278.442 | `dfc2aadda6295f67f677587773f25e5bf8c8ad427ed0db7c518525c91e72e020` |
| `stb_image_write.h` | v1.19 | 80.947 | `5d89edc8bffd848f3e138b2a1dbec67aadbed591894bde16ade7c83dc233af9b` |

- Sumber: `https://raw.githubusercontent.com/nothings/stb/6b5799bae8a1fe3792691ef4b72ae26434c3e799/<berkas>`
  (commit `6b5799bae8a1fe3792691ef4b72ae26434c3e799`), diunduh 22 Nov 2026 atas izin Ridlo.
- Lisensi: dwi-lisensi MIT atau domain publik (Unlicense), pilih salah satu; teksnya ada di akhir
  tiap berkas.
- Hanya dekoder PNG yang dikompilasi (`STBI_ONLY_PNG`).
- Bukan kode proyek: jangan diedit. Untuk mengganti pustaka PNG, tulis backend baru untuk
  `src/cli/png_io.h` lalu `make landcover LC_PNG_SRC=<berkas.cpp> LC_PNG_CHECK=`.
