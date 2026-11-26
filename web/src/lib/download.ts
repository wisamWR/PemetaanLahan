import { ApiError } from '@/lib/api'

/** Nama berkas dari `Content-Disposition` (`filename*=UTF-8''...` dulu, lalu `filename="..."`), atau null. */
export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null
  const star = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(header)
  if (star) {
    try {
      return decodeURIComponent(star[1].trim())
    } catch {
      /* penyandian rusak: coba bentuk biasa */
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(header)
  return plain ? plain[1].trim() : null
}

/**
 * Mengunduh berkas lewat fetch lalu menyimpannya seperti tautan unduhan, supaya kegagalan (404, 500, server mati)
 * bisa ditampilkan sebagai pesan, bukan halaman galat mentah, dan lamanya penyiapan bisa diberi umpan balik.
 * Dipakai untuk PNG resolusi penuh (server menulis berkas sementara dulu, beberapa detik untuk tile 10980²);
 * ekspor kecil tetap tautan `<a download>` biasa. Mengembalikan nama berkas yang dipakai.
 */
export async function downloadFile(url: string, fallbackName: string): Promise<string> {
  let res: Response
  try {
    res = await fetch(url)
  } catch {
    throw new ApiError('Tidak bisa menghubungi API. Pastikan server FastAPI berjalan.', 0)
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* badan bukan JSON: pakai status */
    }
    throw new ApiError(detail, res.status)
  }
  const name = filenameFromDisposition(res.headers.get('content-disposition')) ?? fallbackName
  const objectUrl = URL.createObjectURL(await res.blob())
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = name
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000)
  return name
}
