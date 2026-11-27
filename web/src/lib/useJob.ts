import { useEffect, useRef, useState } from 'react'
import { ApiError, api, type JobSnapshot, type JobStep } from '@/lib/api'

export interface Tick {
  elapsed_s: number
  step: number | null
  step_elapsed_s: number | null
}

const MAX_LINES = 60

/**
 * Mengikuti satu pekerjaan lewat SSE (`/api/jobs/{id}/events`).
 *
 * - Peristiwa penutup `done` / `failed` selalu datang; saat itu koneksi ditutup (tanpa penutupan,
 *   EventSource akan menyambung ulang tanpa henti).
 * - Waktu berjalan (`tick`) hanya progres, bukan hasil: angka kinerja datang dari `timing.json` lewat
 *   `/api/compare`.
 * - Bila koneksi putus permanen (mis. server dimulai ulang), cuplikan diambil lewat `GET /api/jobs/{id}`;
 *   404 berarti pekerjaan hilang dan `onLost` dipanggil.
 */
export function useJob(
  jobId: string | null,
  onFinished: (job: JobSnapshot) => void,
  onLost: () => void,
): { job: JobSnapshot | null; tick: Tick | null } {
  const [job, setJob] = useState<JobSnapshot | null>(null)
  const [tick, setTick] = useState<Tick | null>(null)
  const finished = useRef(onFinished)
  const lost = useRef(onLost)
  finished.current = onFinished
  lost.current = onLost

  useEffect(() => {
    setJob(null)
    setTick(null)
    if (!jobId) return
    let closed = false
    const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`)

    const finish = (snapshot: JobSnapshot) => {
      if (closed) return
      closed = true
      es.close()
      setJob(snapshot)
      setTick(null)
      finished.current(snapshot)
    }
    const parse = <T,>(e: Event): T => JSON.parse((e as MessageEvent<string>).data) as T

    es.addEventListener('snapshot', (e) => setJob(parse<JobSnapshot>(e)))
    es.addEventListener('step', (e) => {
      const step = parse<JobStep>(e)
      setJob((j) => (j ? { ...j, steps: j.steps.map((s) => (s.index === step.index ? step : s)) } : j))
    })
    es.addEventListener('progress', (e) => {
      const { line } = parse<{ step: number; line: string }>(e)
      setJob((j) => (j ? { ...j, lines: [...j.lines, line].slice(-MAX_LINES) } : j))
    })
    es.addEventListener('tick', (e) => setTick(parse<Tick>(e)))
    es.addEventListener('done', (e) => finish(parse<{ job: JobSnapshot }>(e).job))
    es.addEventListener('failed', (e) => finish(parse<{ job: JobSnapshot }>(e).job))
    // 'error' tanpa data = koneksi putus (bukan peristiwa dari server; itu bernama 'failed').
    es.addEventListener('error', () => {
      if (closed || es.readyState !== EventSource.CLOSED) return // masih mencoba menyambung ulang
      api
        .job(jobId)
        .then((snapshot) => (snapshot.status === 'done' || snapshot.status === 'error' ? finish(snapshot) : setJob(snapshot)))
        .catch((err) => {
          if (err instanceof ApiError && err.status === 404 && !closed) {
            closed = true
            lost.current()
          }
        })
    })
    return () => {
      closed = true
      es.close()
    }
  }, [jobId])

  return { job, tick }
}
