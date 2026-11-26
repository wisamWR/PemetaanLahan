// Peta kelas (isian editor) per run, disimpan di browser. localStorage bisa kosong, penuh, atau melempar
// galat (jendela pribadi, data situs diblokir): semua baca/tulis dibungkus try/catch dan halaman tetap
// berfungsi tanpanya (seperti theme.tsx).
//
// Basi: run_id memuat dataset dan K, tetapi run yang ditimpa (Buat peta lagi, parameter sama) memakai
// folder yang sama. Hasil K-Means deterministik, jadi nama kelas tetap berlaku bila hasilnya identik;
// yang berubah hanya bila isi dataset berganti. Karena itu yang dibandingkan adalah SIDIK hasil (counts
// + centroid, dari server), bukan waktu pembuatan: sidik beda = nama dibuang. K diperiksa lagi saat muat.

import type { ClassMapBody } from '@/lib/api'

const PREFIX = 'pemetaan.class_map.v1.'

/** Isian satu cluster: Kelas (pilihan) dan Nama lain (teks bebas); server memutuskan mana yang menang. */
export interface ClusterEntry {
  kelas: string | null
  nama_lain: string
}

export type Rows = Record<number, ClusterEntry>

interface Stored {
  k: number
  fingerprint: string | null
  rows: Record<string, ClusterEntry>
}

export function loadRows(runId: string, k: number): { rows: Rows; fingerprint: string | null } {
  try {
    const raw = window.localStorage.getItem(PREFIX + runId)
    if (!raw) return { rows: {}, fingerprint: null }
    const data = JSON.parse(raw) as Partial<Stored>
    if (data.k !== k || typeof data.rows !== 'object' || data.rows === null) return { rows: {}, fingerprint: null }
    const rows: Rows = {}
    for (const [key, value] of Object.entries(data.rows)) {
      const idx = Number(key)
      if (!Number.isInteger(idx) || idx < 0 || idx >= k || typeof value !== 'object' || value === null) continue
      const kelas = typeof value.kelas === 'string' ? value.kelas : null
      const nama = typeof value.nama_lain === 'string' ? value.nama_lain : ''
      if (kelas !== null || nama !== '') rows[idx] = { kelas, nama_lain: nama }
    }
    return { rows, fingerprint: typeof data.fingerprint === 'string' ? data.fingerprint : null }
  } catch {
    return { rows: {}, fingerprint: null }
  }
}

export function saveRows(runId: string, k: number, fingerprint: string, rows: Rows): void {
  try {
    if (Object.keys(rows).length === 0) {
      window.localStorage.removeItem(PREFIX + runId)
      return
    }
    const stored: Stored = { k, fingerprint, rows }
    window.localStorage.setItem(PREFIX + runId, JSON.stringify(stored))
  } catch {
    /* tanpa penyimpanan: nama kelas hanya bertahan selama halaman terbuka */
  }
}

/** Badan `POST /api/runs/{id}/area`: hanya cluster yang punya isian. */
export function toClassMap(rows: Rows): ClassMapBody {
  const body: ClassMapBody = {}
  for (const [key, entry] of Object.entries(rows)) body[key] = { kelas: entry.kelas, nama_lain: entry.nama_lain }
  return body
}

/** Isian dari tabel cluster server (nama sudah diputuskan di Python), untuk menyusun ulang editor. */
export function isBlank(entry: ClusterEntry | undefined): boolean {
  return !entry || (entry.kelas === null && entry.nama_lain.trim() === '')
}
