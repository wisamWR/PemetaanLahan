import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { LoaderCircle, Map as MapIcon, TriangleAlert } from 'lucide-react'
import { ApiErrorState } from '@/components/ApiErrorState'
import { PageHeader } from '@/components/PageHeader'
import { useToast } from '@/components/Toaster'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { JobPanel } from '@/features/kinerja/JobPanel'
import { ApiError, api, type Config, type JobSnapshot, type RunSummary } from '@/lib/api'
import { useJob } from '@/lib/useJob'
import { ImagePicker } from './ImagePicker'
import { ResultSkeleton, ResultView } from './ResultView'

const PAGE_DESCRIPTION =
  'Peta kelas tutupan lahan dari citra yang dipilih: citra asli berdampingan dengan peta, pelabelan, tabel luas, dan ekspor.'

/**
 * Run yang ditampilkan untuk (citra, K): utamakan run dengan iterasi dan seed bawaan (yang dibuat tombol Buat
 * peta), lalu yang terbaru dengan mode Optimalisasi, selain itu yang terbaru. Run dari halaman Kinerja dengan
 * parameter yang sama boleh dipakai (CPU saja dan Optimalisasi sebanding, hasilnya identik bila verifikasi lulus);
 * run dengan iterasi atau seed lain hanya dipakai bila tidak ada yang bawaan.
 */
function pickRun(runs: RunSummary[], dataset: string, k: number, defaults: Config['defaults']): RunSummary | undefined {
  const matches = runs.filter((r) => r.dataset === dataset && r.k === k)   // `runs` sudah terbaru dulu
  const standard = matches.filter((r) => r.iters === defaults.iters && r.seed === defaults.seed)
  const pool = standard.length > 0 ? standard : matches
  return pool.find((r) => r.mode === 'opt') ?? pool[0]
}

function PickerSkeleton() {
  return (
    <Card aria-hidden>
      <CardHeader>
        <Skeleton className="h-4 w-16" />
        <Skeleton className="mt-1 h-3.5 w-80 max-w-full" />
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="flex flex-col gap-2 rounded-xl border p-2">
            <Skeleton className="aspect-[4/3] w-full" />
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-3 w-20" />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

/** Ilustrasi kosong: citra → peta kelas (pola Railway dan Steep: gambar ringan di atas judul). */
function EmptyIllustration() {
  return (
    <svg viewBox="0 0 208 88" className="h-20 w-52 text-muted-foreground" fill="none" aria-hidden>
      <rect x="1" y="1" width="206" height="86" rx="10" stroke="currentColor" strokeOpacity=".35" strokeDasharray="4 4" />
      <rect x="16" y="16" width="62" height="56" rx="6" fill="currentColor" fillOpacity=".18" />
      <path d="M22 62 L40 40 L52 54 L62 44 L72 62 Z" fill="currentColor" fillOpacity=".4" />
      <circle cx="60" cy="28" r="5" fill="currentColor" fillOpacity=".4" />
      <path d="M92 44 h24 m-7 -7 l7 7 l-7 7" stroke="currentColor" strokeOpacity=".6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="130" y="16" width="62" height="56" rx="6" fill="currentColor" fillOpacity=".1" />
      <rect x="130" y="16" width="30" height="28" rx="6" fill="currentColor" fillOpacity=".32" />
      <rect x="160" y="44" width="32" height="28" rx="6" fill="currentColor" fillOpacity=".32" />
      <rect x="160" y="16" width="32" height="28" fill="currentColor" fillOpacity=".18" />
    </svg>
  )
}

/** Tiga langkah pemakaian (pola daftar "Mulai" Eventbrite dan Slite): angka bulat, judul, keterangan redup. */
const QUICK_START = [
  { title: 'Pilih citra', hint: 'Kartu bergambar di atas, atau unggah PNG sendiri.' },
  { title: 'Atur jumlah cluster (K), lalu Buat peta', hint: 'K-Means dijalankan dengan mode Optimalisasi; progresnya tampil di bawah.' },
  { title: 'Beri nama cluster, lihat luas, ekspor', hint: 'Legenda sekaligus editor nama kelas; tabel luas dalam persen dan hektare.' },
]

function EmptyState({
  dataset,
  k,
  disabled,
  busy,
  firstUse,
  onMake,
}: {
  dataset: string
  k: number
  disabled: boolean
  busy: boolean
  /** Belum ada satu pun hasil di disk: tampilkan langkah mulai cepat (pengenalan ringan, hilang sendiri setelah run pertama). */
  firstUse: boolean
  onMake: () => void
}) {
  return (
    <Card className="border-dashed" data-testid="empty-state">
      <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
        <EmptyIllustration />
        <h2 className="text-lg font-semibold">
          {dataset ? `Belum ada peta untuk ${dataset} dengan K=${k}` : 'Belum ada citra'}
        </h2>
        <p className="max-w-md text-sm text-muted-foreground">
          {dataset ? (
            <>
              Tekan <b>Buat peta</b> untuk menjalankan K-Means pada citra ini. Citra asli berdampingan dengan peta kelas,
              legenda untuk memberi nama cluster, dan tabel luas muncul di sini setelah run selesai.
            </>
          ) : (
            'Unggah citra PNG, atau taruh folder tile (dengan manifest.json) atau PNG di data/, lalu muat ulang halaman.'
          )}
        </p>
        {firstUse && dataset && (
          <ol className="mt-1 flex w-full max-w-md flex-col gap-3 text-left" data-testid="quick-start" aria-label="Mulai cepat">
            {QUICK_START.map((step, i) => (
              <li key={step.title} className="flex items-start gap-3">
                <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border text-xs font-medium tabular-nums text-muted-foreground">
                  {i + 1}
                </span>
                <span className="flex flex-col gap-0.5">
                  <span className="text-sm font-medium">{step.title}</span>
                  <span className="text-xs text-muted-foreground">{step.hint}</span>
                </span>
              </li>
            ))}
          </ol>
        )}
        {/* Di layar sempit tombol yang sama sudah ada di bilah tetap di dasar layar (PageHeader). */}
        <Button className="mt-1 max-md:hidden" disabled={disabled} onClick={onMake}>
          {busy ? <LoaderCircle className="animate-spin" /> : <MapIcon />}
          {busy ? 'Membuat peta…' : 'Buat peta'}
        </Button>
      </CardContent>
    </Card>
  )
}

function MakeMapButton({ disabled, busy, onMake }: { disabled: boolean; busy: boolean; onMake: () => void }) {
  return (
    <Button disabled={disabled} onClick={onMake} data-testid="make-map" aria-busy={busy || undefined}>
      {busy ? <LoaderCircle className="animate-spin" /> : <MapIcon />}
      {busy ? 'Membuat peta…' : 'Buat peta'}
    </Button>
  )
}

export function PemetaanPage() {
  const qc = useQueryClient()
  const toast = useToast()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs })

  const [dataset, setDataset] = useState<string | null>(null)
  const [k, setK] = useState<number | null>(null)
  const [pin, setPin] = useState<{ dataset: string; k: number; runId: string } | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [formError, setFormError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [resumed, setResumed] = useState(false)

  // Isian awal: citra pertama dan K bawaan server.
  useEffect(() => {
    if (!config.data || !datasets.data) return
    setK((cur) => cur ?? config.data.defaults.k)
    setDataset((cur) => cur ?? datasets.data.datasets[0]?.name ?? null)
  }, [config.data, datasets.data])

  // Muat ulang halaman saat run masih berjalan: sambung lagi ke pekerjaan aktif (slot satu pekerjaan dipakai bersama Kinerja).
  useEffect(() => {
    if (resumed || !config.data) return
    setResumed(true)
    if (config.data.active_job) setJobId(config.data.active_job)
  }, [resumed, config.data])

  const onFinished = useCallback(
    (job: JobSnapshot) => {
      for (const key of ['runs', 'compare', 'verify', 'config']) qc.invalidateQueries({ queryKey: [key] })
      const done = job.steps.find((s) => s.status === 'done')
      if (done && dataset !== null && k !== null) setPin({ dataset, k, runId: done.run_id })
      // Toast supaya selesainya terlihat walau layar tergulir ke tempat lain (rincian galat tetap di panel progres).
      if (job.status === 'done') {
        toast.show({ kind: 'success', title: 'Peta selesai dibuat', description: dataset !== null && k !== null ? `${dataset}, K=${k}` : undefined })
      } else {
        toast.show({ kind: 'error', title: 'Peta gagal dibuat', description: job.error?.message })
      }
    },
    [qc, dataset, k, toast],
  )
  const onLost = useCallback(() => setJobId(null), [])
  const { job, tick } = useJob(jobId, onFinished, onLost)
  const running = job != null && (job.status === 'running' || job.status === 'queued')

  const start = useMutation({
    mutationFn: async () => {
      const cfg = config.data as Config
      if (dataset === null || k === null) throw new Error('Pilih citra dulu.')
      const d = cfg.defaults
      return api.startRun({ dataset, mode: 'opt', kmeans: d.kmeans, k, iters: d.iters, seed: d.seed, threads: d.threads, force: false })
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

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadDataset(file),
    onMutate: () => setUploadError(null),
    onSuccess: (card) => {
      setDataset(card.name)
      qc.invalidateQueries({ queryKey: ['datasets'] })
    },
    onError: (e) => setUploadError(e.message),
  })

  if (config.isError) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Pemetaan" description={PAGE_DESCRIPTION} />
        <ApiErrorState message={config.error.message} onRetry={() => config.refetch()} />
      </div>
    )
  }
  if (datasets.isError) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Pemetaan" description={PAGE_DESCRIPTION} />
        <ApiErrorState message={datasets.error.message} onRetry={() => datasets.refetch()} />
      </div>
    )
  }
  if (!config.data || !datasets.data || k === null) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Pemetaan" description={PAGE_DESCRIPTION} />
        <PickerSkeleton />
        <ResultSkeleton />
      </div>
    )
  }

  const cfg = config.data
  const cards = datasets.data.datasets
  const selected = dataset ?? ''
  const busy = running || start.isPending
  const runDisabled = busy || upload.isPending || selected === ''
  const onMake = () => start.mutate()

  const pinned = pin && pin.dataset === selected && pin.k === k ? pin.runId : null
  const run =
    runs.data == null
      ? undefined
      : (runs.data.runs.find((r) => r.run_id === pinned) ?? pickRun(runs.data.runs, selected, k, cfg.defaults))

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Pemetaan" description={PAGE_DESCRIPTION} actions={<MakeMapButton disabled={runDisabled} busy={busy} onMake={onMake} />} />

      <ImagePicker
        config={cfg}
        datasets={cards}
        uploadHint={datasets.data.upload.hint}
        dataset={selected}
        k={k}
        uploading={upload.isPending}
        uploadError={uploadError}
        onDataset={(name) => {
          setDataset(name)
          if (!running) setJobId(null)       // panel progres run sebelumnya (selesai atau galat) tidak relevan untuk citra baru
        }}
        onK={(value) => {
          setK(value)
          if (!running) setJobId(null)
        }}
        onUpload={(file) => upload.mutate(file)}
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

      {runs.isError ? (
        <ApiErrorState message={runs.error.message} onRetry={() => runs.refetch()} />
      ) : !runs.data ? (
        <ResultSkeleton />
      ) : run ? (
        <ResultView key={run.run_id} run={run} />
      ) : (
        <EmptyState dataset={selected} k={k} disabled={runDisabled} busy={busy} firstUse={runs.data.runs.length === 0} onMake={onMake} />
      )}
    </div>
  )
}
