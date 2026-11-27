import { useEffect, useState } from 'react'

/** Nilai yang "menyusul" `value` setelah `ms` tanpa perubahan (debounce suntingan nama kelas). */
export function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return debounced
}

export interface ImageState {
  /** URL yang SUDAH selesai dimuat (gambar sebelumnya dipertahankan sampai yang baru siap: tanpa kedip). */
  src: string | null
  status: 'loading' | 'ok' | 'error'
  /** Alasan dari server bila gambar tidak ada (mis. "Citra asli tidak ditemukan ..."). */
  reason: string | null
}

/**
 * Muat gambar di latar dan ganti `src` hanya setelah siap. Bila gagal, alasan dibaca dari badan JSON galat
 * server (`{"detail": "..."}`) supaya UI bisa menampilkannya, bukan sekadar ikon rusak.
 */
export function useImage(url: string | null): ImageState {
  const [state, setState] = useState<ImageState>({ src: null, status: 'loading', reason: null })
  useEffect(() => {
    if (!url) return
    let alive = true
    const img = new Image()
    img.onload = () => alive && setState({ src: url, status: 'ok', reason: null })
    img.onerror = () => {
      fetch(url)
        .then((res) => res.json())
        .then((body) => (typeof body?.detail === 'string' ? body.detail : 'Gambar tidak bisa dimuat.'))
        .catch(() => 'Gambar tidak bisa dimuat.')
        .then((reason) => alive && setState((s) => ({ src: s.src, status: 'error', reason })))
    }
    img.src = url
    return () => {
      alive = false
      img.onload = null
      img.onerror = null
    }
  }, [url])
  return state
}
