import { useCallback, useEffect, useRef, useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { TriangleAlert } from 'lucide-react'
import { KpiCell, KpiStrip } from '@/components/KpiStrip'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { api, type RunSummary } from '@/lib/api'
import { fmtInt } from '@/lib/format'
import { loadRows, saveRows, toClassMap, type ClusterEntry, type Rows } from '@/lib/labelStore'
import { cn } from '@/lib/utils'
import { AreaCard } from './AreaCard'
import { useDebounced } from './hooks'
import { LegendCard } from './LegendCard'
import { MapCard } from './MapCard'

const EDIT_DEBOUNCE_MS = 180

/** Kerangka memuat yang meniru susunan hasil: strip KPI, peta + legenda, tabel (Square, Neon, Linktree). */
export function ResultSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-label="Memuat hasil" data-testid="result-skeleton">
      <Card className="gap-0 py-0">
        <div className="grid grid-cols-2 gap-px bg-border xl:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className={cn('flex flex-col gap-2.5 bg-card px-5 py-4', (i === 0 || i === 3) && 'col-span-2 sm:col-span-1')}>
              <Skeleton className="h-3.5 w-24" />
              <Skeleton className="h-8 w-28" />
              <Skeleton className="h-3 w-32 max-w-full" />
            </div>
          ))}
        </div>
      </Card>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(340px,420px)]">
        <Card>
          <CardHeader>
            <Skeleton className="h-4 w-44" />
            <Skeleton className="mt-1 h-3.5 w-72 max-w-full" />
          </CardHeader>
          <CardContent>
            <Skeleton className="aspect-square w-full max-w-[560px]" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-4 w-40" />
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="flex flex-col gap-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-8 w-full" />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-4 w-32" />
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-5 w-full" />
          ))}
        </CardContent>
      </Card>
    </div>
  )
}

function contextLine(run: RunSummary): string {
  return `${run.dataset} · ${run.mode_label} · K=${run.k} · ${run.iters} iterasi · seed ${run.seed} · ${fmtInt(run.width)}×${fmtInt(run.height)} (${fmtInt(run.n_pixels)} piksel) · ${run.created_text}`
}

/**
 * Hasil satu run: strip KPI, peta, legenda/pelabelan, tabel luas. Dipasang dengan `key={run_id}`, jadi pergantian
 * run mengatur ulang isian. Isian nama kelas ada di state (sumber kebenaran untuk kolom Kelas/Nama lain),
 * dikirim ke `POST /api/runs/{id}/area` setelah jeda singkat; server mengembalikan nama akhir, palet, dan tabel.
 * Isian disimpan di localStorage per run (labelStore) dan dibuang bila sidik hasil run berubah.
 */
export function ResultView({ run }: { run: RunSummary }) {
  const initial = useRef(loadRows(run.run_id, run.k))
  const [rows, setRows] = useState<Rows>(() => initial.current.rows)
  const debounced = useDebounced(rows, EDIT_DEBOUNCE_MS)
  const lastFingerprint = useRef<string | null>(initial.current.fingerprint)

  const area = useQuery({
    queryKey: ['area', run.run_id, run.created_at, JSON.stringify(debounced)],
    queryFn: () => api.area(run.run_id, toClassMap(debounced)),
    placeholderData: keepPreviousData,
    retry: false,
  })

  useEffect(() => {
    const data = area.data
    if (!data || area.isPlaceholderData) return
    if (lastFingerprint.current !== null && lastFingerprint.current !== data.fingerprint) {
      // Hasil run berubah (dataset diganti, atau run ditimpa dengan hasil lain): nama lama tidak berlaku.
      lastFingerprint.current = data.fingerprint
      saveRows(run.run_id, run.k, data.fingerprint, {})
      setRows({})
      return
    }
    lastFingerprint.current = data.fingerprint
    saveRows(run.run_id, run.k, data.fingerprint, debounced)
  }, [area.data, area.isPlaceholderData, debounced, run.run_id, run.k])

  const onChange = useCallback((cluster: number, patch: Partial<ClusterEntry>) => {
    setRows((prev) => {
      const base: ClusterEntry = prev[cluster] ?? { kelas: null, nama_lain: '' }
      const next: ClusterEntry = { ...base, ...patch }
      const rest = { ...prev }
      if (next.kelas === null && next.nama_lain === '') delete rest[cluster]
      else rest[cluster] = next
      return rest
    })
  }, [])

  if (area.isError && !area.data) {
    return (
      <Alert variant="destructive" data-testid="area-error">
        <TriangleAlert />
        <AlertDescription>
          <p>{area.error.message}</p>
          <Button variant="outline" size="sm" className="mt-2" onClick={() => area.refetch()}>
            Coba lagi
          </Button>
        </AlertDescription>
      </Alert>
    )
  }
  if (!area.data) return <ResultSkeleton />
  const data = area.data

  return (
    <section className="flex flex-col gap-4" aria-label="Hasil pemetaan" data-testid="result-view">
      {run.problems.length > 0 && (
        <Alert variant="warning">
          <TriangleAlert />
          <AlertDescription>
            <ul className="list-disc pl-4">
              {run.problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      <KpiStrip
        label="Ringkasan luas"
        gridClassName="grid grid-cols-2 gap-px bg-border xl:grid-cols-4"
        footer={contextLine(run)}
      >
        {/* 2×2 (4 sel sebaris di layar lebar); di layar sempit sel pertama dan terakhir selebar baris, jadi
            nilai panjang ("1,205,604.00 ha") muat dan strip tidak menjulang. */}
        {data.kpis.map((kpi, i) => (
          <KpiCell
            key={kpi.key}
            testId={`kpi-${kpi.key}`}
            title={kpi.label}
            accent={i === 0 ? 'var(--foreground)' : 'color-mix(in oklab, var(--foreground) 28%, transparent)'}
            value={kpi.value}
            caption={kpi.caption}
            mock={i === 0 && data.mock}
            valueSize="auto"
            className={cn('bg-card', (i === 0 || i === data.kpis.length - 1) && 'col-span-2 sm:col-span-1')}
          />
        ))}
      </KpiStrip>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(340px,420px)] xl:items-start">
        <MapCard area={data} />
        <LegendCard area={data} rows={rows} onChange={onChange} />
      </div>
      <AreaCard area={data} />
    </section>
  )
}
