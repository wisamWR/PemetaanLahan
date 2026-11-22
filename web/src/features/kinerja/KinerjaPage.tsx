import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { TriangleAlert } from 'lucide-react'
import { ApiErrorState } from '@/components/ApiErrorState'
import { PageHeader } from '@/components/PageHeader'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ApiError, api, type Config, type JobSnapshot, type Mode, type PairSelection, type RunRequest } from '@/lib/api'
import { useJob } from '@/lib/useJob'
import { JobPanel } from './JobPanel'
import { KpiRow } from './KpiRow'
import { PairBar } from './PairBar'
import { ParamsCard, RunActions, type FormState } from './ParamsCard'
import { StageChartCard } from './StageChartCard'
import { VerifyCard } from './VerifyCard'

const PAGE_DESCRIPTION =
  'Bandingkan CPU saja dengan Optimalisasi: waktu per tahap, speedup, dan status verifikasi antarmode.'

function toInt(label: string, raw: string): number {
  const n = Number(raw)
  if (raw.trim() === '' || !Number.isInteger(n)) throw new Error(`${label} harus bilangan bulat.`)
  return n
}

/** Kerangka memuat yang meniru susunan halaman (Square, Neon, Linktree): kartu parameter, strip KPI, grafik + tabel. */
function ParamsSkeleton() {
  return (
    <Card size="sm" aria-hidden>
      <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-[1.3fr_1.2fr_0.7fr_1.2fr]">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="flex flex-col gap-2">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-9 w-full" />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function ResultsSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-label="Memuat hasil">
      <div className="flex items-center justify-between gap-3">
        <Skeleton className="h-5 w-16" />
        <Skeleton className="h-9 w-64 max-w-full" />
      </div>
      <Card className="gap-0 py-0">
        <div className="grid divide-y md:grid-cols-3 md:divide-x md:divide-y-0">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex flex-col gap-2.5 px-5 py-4">
              <Skeleton className="h-3.5 w-28" />
              <Skeleton className="h-8 w-24" />
              <Skeleton className="h-3 w-40 max-w-full" />
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <CardHeader>
          <Skeleton className="h-4 w-36" />
          <Skeleton className="mt-1 h-3.5 w-80 max-w-full" />
        </CardHeader>
        <CardContent className="grid gap-6 xl:grid-cols-2">
          <Skeleton className="h-44 w-full" />
          <div className="flex flex-col gap-3">
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-5 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

/** Ilustrasi kosong: grafik batang tahap yang diredupkan (pola Railway dan Steep: gambar ringan di atas judul). */
function EmptyIllustration() {
  return (
    <svg viewBox="0 0 176 88" className="h-20 w-44 text-muted-foreground" fill="none" aria-hidden>
      <rect x="1" y="1" width="174" height="86" rx="10" stroke="currentColor" strokeOpacity=".35" strokeDasharray="4 4" />
      <rect x="16" y="22" width="26" height="12" rx="3" fill="currentColor" fillOpacity=".3" />
      <rect x="44" y="22" width="86" height="12" rx="3" fill="currentColor" fillOpacity=".45" />
      <rect x="132" y="22" width="28" height="12" rx="3" fill="currentColor" fillOpacity=".3" />
      <rect x="16" y="52" width="14" height="12" rx="3" fill="currentColor" fillOpacity=".3" />
      <rect x="32" y="52" width="10" height="12" rx="3" fill="currentColor" fillOpacity=".45" />
      <rect x="44" y="52" width="20" height="12" rx="3" fill="currentColor" fillOpacity=".3" />
    </svg>
  )
}

function EmptyState({
  config,
  mode,
  disabled,
  onRun,
  onCompare,
}: {
  config: Config
  mode: Mode
  disabled: boolean
  onRun: () => void
  onCompare: () => void
}) {
  return (
    <Card className="border-dashed" data-testid="empty-state">
      <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
        <EmptyIllustration />
        <h2 className="text-lg font-semibold">Belum ada hasil</h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Pilih dataset, lalu tekan <b>Bandingkan</b> untuk menjalankan CPU saja dan Optimalisasi. Kartu speedup, grafik waktu
          per tahap, dan status verifikasi muncul di sini setelah kedua run selesai.
        </p>
        <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
          <RunActions config={config} mode={mode} disabled={disabled} onRun={onRun} onCompare={onCompare} />
        </div>
      </CardContent>
    </Card>
  )
}

export function KinerjaPage() {
  const qc = useQueryClient()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs })

  const [form, setForm] = useState<FormState | null>(null)
  const [selection, setSelection] = useState<PairSelection>({})
  const [jobId, setJobId] = useState<string | null>(null)
  const [version, setVersion] = useState(0)
  const [formError, setFormError] = useState<string | null>(null)
  const [resumed, setResumed] = useState(false)

  // Isian awal dari konfigurasi server dan dataset pertama.
  useEffect(() => {
    if (form || !config.data || !datasets.data) return
    const d = config.data.defaults
    setForm({
      dataset: datasets.data.datasets[0]?.name ?? '',
      k: d.k,
      iters: String(d.iters),
      seed: String(d.seed),
      threads: d.threads,
      kmeans: d.kmeans,
      mode: d.mode,
      force: false,
    })
  }, [form, config.data, datasets.data])

  // Muat ulang halaman saat run masih berjalan: sambung lagi ke pekerjaan aktif.
  useEffect(() => {
    if (resumed || !config.data) return
    setResumed(true)
    if (config.data.active_job) setJobId(config.data.active_job)
  }, [resumed, config.data])

  const onFinished = useCallback(
    (job: JobSnapshot) => {
      for (const key of ['runs', 'compare', 'verify', 'config']) qc.invalidateQueries({ queryKey: [key] })
      const done = job.steps.filter((s) => s.status === 'done')
      const cpu = done.find((s) => s.mode === 'cpu')
      const opt = done.find((s) => s.mode === 'opt')
      if (cpu || opt) setSelection({ cpu: cpu?.run_id, opt: opt?.run_id })
      setVersion((v) => v + 1)
    },
    [qc],
  )
  const onLost = useCallback(() => setJobId(null), [])
  const { job, tick } = useJob(jobId, onFinished, onLost)
  const running = job != null && (job.status === 'running' || job.status === 'queued')

  const start = useMutation({
    mutationFn: async (kind: 'run' | 'compare') => {
      if (!form) throw new Error('Formulir belum siap.')
      const body: Omit<RunRequest, 'mode'> = {
        dataset: form.dataset,
        kmeans: form.kmeans,
        k: form.k,
        iters: toInt('Iterasi', form.iters),
        seed: toInt('Seed', form.seed),
        threads: form.threads,
        force: form.force,
      }
      return kind === 'run' ? api.startRun({ ...body, mode: form.mode }) : api.startCompare(body)
    },
    onMutate: () => setFormError(null),
    onSuccess: (d) => {
      setJobId(d.job_id)
      qc.invalidateQueries({ queryKey: ['config'] })
    },
    onError: (e) => {
      if (e instanceof ApiError && e.status === 409 && e.jobId) setJobId(e.jobId)
      setFormError(e.message)
    },
  })

  const compare = useQuery({
    queryKey: ['compare', selection],
    queryFn: () => api.compare(selection),
    enabled: !!config.data,
    placeholderData: (prev) => prev,
  })

  // Sebelum parameter siap (atau saat API galat), kepala halaman tetap tampil tanpa tombol aksi.
  if (config.isError) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Kinerja" description={PAGE_DESCRIPTION} />
        <ApiErrorState message={config.error.message} onRetry={() => config.refetch()} />
      </div>
    )
  }
  if (!config.data || !datasets.data || !form) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Kinerja" description={PAGE_DESCRIPTION} />
        {datasets.isError ? (
          <Alert variant="destructive">
            <AlertDescription>{datasets.error.message}</AlertDescription>
          </Alert>
        ) : (
          <>
            <ParamsSkeleton />
            <ResultsSkeleton />
          </>
        )}
      </div>
    )
  }

  const cfg = config.data
  const data = compare.data
  const problems = [...(data?.cpu?.problems ?? []), ...(data?.opt?.problems ?? [])]
  const runDisabled = running || start.isPending || datasets.data.datasets.length === 0
  const onRun = () => start.mutate('run')
  const onCompare = () => start.mutate('compare')

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Kinerja"
        description={PAGE_DESCRIPTION}
        actions={<RunActions config={cfg} mode={form.mode} disabled={runDisabled} onRun={onRun} onCompare={onCompare} />}
      />

      <ParamsCard
        config={cfg}
        datasets={datasets.data.datasets}
        form={form}
        onChange={(patch) => setForm({ ...form, ...patch })}
      />

      {cfg.core && !cfg.core.ok && (
        <Alert variant="warning" data-testid="core-problem">
          <TriangleAlert />
          <AlertTitle>Program landcover belum bisa dipakai</AlertTitle>
          <AlertDescription>{cfg.core.problem}</AlertDescription>
        </Alert>
      )}
      {formError && (
        <Alert variant="destructive" data-testid="form-error">
          <TriangleAlert />
          <AlertDescription>{formError}</AlertDescription>
        </Alert>
      )}
      {job && <JobPanel job={job} tick={tick} onDismiss={() => setJobId(null)} />}
      {datasets.data.datasets.length === 0 && (
        <Alert variant="info">
          <AlertDescription>
            Tidak ada dataset. Taruh folder tile (dengan manifest.json) atau PNG di data/, lalu muat ulang halaman.
          </AlertDescription>
        </Alert>
      )}
      {(runs.data?.unreadable.length ?? 0) > 0 && (
        <Alert variant="warning">
          <TriangleAlert />
          <AlertDescription>
            {runs.data!.unreadable.map((u) => (
              <p key={u.run_id}>
                Hasil {u.run_id} tidak terbaca: {u.reason}
              </p>
            ))}
          </AlertDescription>
        </Alert>
      )}

      {compare.isError && (
        <Alert variant="destructive">
          <TriangleAlert />
          <AlertDescription>
            <p>{compare.error.message}</p>
            <Button variant="outline" size="sm" className="mt-2" onClick={() => setSelection({})}>
              Tampilkan hasil terbaru
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {data?.empty ? (
        <EmptyState config={cfg} mode={form.mode} disabled={runDisabled} onRun={onRun} onCompare={onCompare} />
      ) : data ? (
        <section className="flex flex-col gap-4" aria-label="Hasil">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-base font-semibold tracking-tight">Hasil</h2>
            {runs.data && <PairBar groups={runs.data.groups} runs={runs.data.runs} data={data} onSelect={setSelection} />}
          </div>
          {problems.length > 0 && (
            <Alert variant="warning">
              <TriangleAlert />
              <AlertTitle>Hasil ini terbaca tetapi janggal</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-4">
                  {problems.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
          <KpiRow data={data} config={cfg} />
          {data.chart && <StageChartCard chart={data.chart} />}
          <VerifyCard data={data} version={String(version)} />
        </section>
      ) : (
        !compare.isError && <ResultsSkeleton />
      )}
    </div>
  )
}
