// Klien API dan tipe respons. Semua kalimat dan angka tampilan (vonis verifikasi, persen terpotong,
// durasi, speedup) datang dari server: frontend hanya menampilkan, tidak menghitung ulang aturan.

export type Level = 'ok' | 'partial' | 'bad' | 'warn'
export type Mode = 'cpu' | 'opt'
export type KmeansChoice = 'auto' | 'omp' | 'cuda'

export interface CoreInfo {
  ok: boolean
  problem: string | null
  info: {
    cpu: string | null
    omp: boolean
    cuda: boolean
    openmp: boolean
    max_threads: number
    auto_cuda_threshold_pixels: number | null
  } | null
  notice: string | null
  kmeans_choices: KmeansChoice[]
}

export interface Config {
  backend: 'mock' | 'cli'
  mock: boolean
  backend_label: string
  mock_notice: string | null
  core: CoreInfo | null
  defaults: { k: number; iters: number; seed: number; threads: number; mode: Mode; kmeans: KmeansChoice }
  limits: { k_min: number; k_max: number; iters_max: number; threads_min: number; threads_max: number; seed_max: number }
  modes: Mode[]
  labels: { modes: Record<string, string>; kmeans: Record<string, string>; devices: Record<string, string>; stages: Record<string, string> }
  device_colors: Record<string, string>
  unknown_device_color: string
  stage_order: string[]
  active_job: string | null
}

export interface Dataset {
  name: string
  label: string
  kind: string
}

/** Kartu citra (`GET /api/datasets`): ukuran dan keterangan datang dari server. */
export interface DatasetCard extends Dataset {
  width: number | null
  height: number | null
  size_text: string
  mock: boolean
  has_preview: boolean
}

export interface UploadLimits {
  max_bytes: number
  max_pixels: number
  hint: string
}

export interface DatasetsResponse {
  datasets: DatasetCard[]
  skipped: string[]
  upload: UploadLimits
}

export interface RunSummary {
  run_id: string
  dataset: string
  mode: Mode
  mode_label: string
  kmeans_device: string
  engine_label: string
  caption: string
  k: number
  iters: number
  seed: number
  threads: number | null
  width: number
  height: number
  n_pixels: number
  total_s: number | null
  total_text: string
  created_at: string | null
  created_text: string
  mock: boolean
  has_labels: boolean
  problems: string[]
  note: string | null
}

export interface Group {
  dataset: string
  k: number
  iters: number
  seed: number
  cpu_run_id: string | null
  opt_run_ids: string[]
  latest_run_id: string
  latest_ts: number
  complete: boolean
}

export interface RunsResponse {
  runs: RunSummary[]
  unreadable: { run_id: string; reason: string }[]
  groups: Group[]
}

export interface StageCell {
  stage: string
  name: string
  device: string
  device_name: string
  color: string
  seconds: number
  duration_text: string
  share: number
  share_text: string
  detail: { key: string; seconds: number; text: string }[]
}

export interface ChartRow {
  label: string
  run_id: string
  mode: Mode
  total_s: number | null
  total_text: string | null
  sum_s: number
  sum_text: string
  stages: StageCell[]
}

export interface Chart {
  labels: string[]
  rows: ChartRow[]
  stage_speedup: Record<string, number | null> | null
  stage_speedup_text: Record<string, string> | null
  stages: { stage: string; name: string }[]
  devices: { device: string; name: string; color: string }[]
  max_seconds: number
  total_name: string
  table: { run: string; stage: string; device: string | null; seconds: number | null; percent: number | null; is_total: boolean }[]
  mock: boolean
}

export interface Compare {
  empty: boolean
  key: { dataset: string; k: number; iters: number; seed: number } | null
  cpu: RunSummary | null
  opt: RunSummary | null
  speedup: number | null
  speedup_text: string
  mock: boolean
  mock_notice: string | null
  chart: Chart | null
}

export interface Verification {
  title: string
  status: 'identical' | 'different' | 'not_comparable' | 'error'
  level: Level
  icon: string
  verdict: string
  source: string | null
  source_text: string | null
  full: boolean
  n_compared: number | null
  n_diff: number | null
  identical_pct: number | null
  identical_pct_text: string | null
  centroids_equal: boolean | null
  counts_equal: boolean | null
  reasons: string[]
  differences: string[]
  notes: string[]
  rows: { what: string; result: string }[]
  pair_label: string | null
  mock: boolean
  mock_notice: string | null
}

export type StepStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped'

export interface JobStep {
  index: number
  label: string
  mode: Mode
  run_id: string
  status: StepStatus
  note: string | null
  reused: boolean
  seconds: number | null
}

export interface JobSnapshot {
  id: string
  kind: 'run' | 'compare'
  status: 'queued' | 'running' | 'done' | 'error'
  force: boolean
  created_at: string
  elapsed_s: number
  running_step: number | null
  steps: JobStep[]
  lines: string[]
  error: { message: string; detail: string | null; step: number } | null
  run_ids: string[]
}

export interface RunRequest {
  dataset: string
  mode?: Mode
  kmeans: KmeansChoice
  k: number
  iters: number
  seed: number
  threads: number
  force: boolean
}

export interface JobStarted {
  job_id: string
  run_ids: string[]
  status: string
}

// --- Pemetaan: luas, pelabelan, ekspor -------------------------------------------------------------

export type ExportKind = 'map' | 'area' | 'clusters' | 'full'

export interface ClusterRow {
  cluster: number
  centroid_hex: string
  class_hex: string
  name: string
  named: boolean
  kelas: string | null
  nama_lain: string
  pixels: number
  pixels_text: string
  percent: number
  percent_text: string
  area_ha: number
  area_text: string
  invalid: boolean
  all_nodata: boolean
  nodata_pixels: number
  nodata_badge: string | null
}

export interface AreaRow {
  name: string
  clusters: string
  cluster_ids: number[]
  colors: string[]
  unnamed: boolean
  is_total: boolean
  is_nodata: boolean
  pixels: number
  pixels_text: string
  percent: number
  percent_text: string
  area_ha: number
  area_text: string
}

export interface KpiItem {
  key: string
  label: string
  value: string
  caption: string
}

export interface ExportItem {
  kind: ExportKind
  label: string
  filename: string
  media_type: string
  available: boolean
  reason: string | null
  hint: string | null
}

export interface AreaResponse {
  run_id: string
  mock: boolean
  mock_notice: string | null
  k: number
  fingerprint: string
  stamp: string
  created_at: string | null
  preview: { width: number; height: number }
  pixel_size_m: number
  pixel_size_note: string | null
  presets: string[]
  unnamed_label: string
  total_label: string
  nodata_label: string
  nodata: {
    pixels: number
    cluster: number
    percent: number | null
    note: string | null
  }
  name_max_chars: number
  names: string[]
  palette: string[]
  clusters: ClusterRow[]
  area: AreaRow[]
  kpis: KpiItem[]
  errors: string[]
  help: { panel: string; other: string; kelas: string }
  preview_caption: string
  area_caption: string
  exports: ExportItem[]
  export_note: string | null
}

/** Isian editor satu cluster; aturan "Nama lain mengalahkan Kelas" diterapkan server. */
export type ClassMapBody = Record<string, { kelas: string | null; nama_lain: string | null }>

export type MapView = 'class' | 'centroid'

export class ApiError extends Error {
  status: number
  jobId?: string
  constructor(message: string, status: number, jobId?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.jobId = jobId
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError('Tidak bisa menghubungi API. Pastikan server FastAPI berjalan (uvicorn api.main:app).', 0)
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    let jobId: string | undefined
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      if (typeof body.job_id === 'string') jobId = body.job_id
    } catch {
      /* badan bukan JSON: pakai status */
    }
    throw new ApiError(detail, res.status, jobId)
  }
  return (await res.json()) as T
}

function query(params: Record<string, string | number | undefined | null>): string {
  const usp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') usp.set(k, String(v))
  const s = usp.toString()
  return s ? `?${s}` : ''
}

export interface PairSelection {
  key?: { dataset: string; k: number; iters: number; seed: number }
  cpu?: string
  opt?: string
}

export const api = {
  config: () => request<Config>('/api/config'),
  datasets: () => request<DatasetsResponse>('/api/datasets'),
  /** Unggah PNG: badan mentah + ?filename= (bukan multipart; python-multipart tidak termasuk stack). */
  uploadDataset: (file: File) =>
    request<DatasetCard>(`/api/datasets${query({ filename: file.name })}`, {
      method: 'POST',
      headers: { 'Content-Type': 'image/png' },
      body: file,
    }),
  area: (runId: string, classMap: ClassMapBody) =>
    request<AreaResponse>(`/api/runs/${encodeURIComponent(runId)}/area`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ class_map: classMap }),
    }),
  runs: () => request<RunsResponse>('/api/runs'),
  compare: (sel: PairSelection) =>
    request<Compare>(
      `/api/compare${query({
        dataset: sel.key?.dataset,
        k: sel.key?.k,
        iters: sel.key?.iters,
        seed: sel.key?.seed,
        cpu: sel.cpu,
        opt: sel.opt,
      })}`,
    ),
  verify: (cpu: string, opt: string) => request<Verification>(`/api/verify${query({ cpu, opt })}`),
  job: (id: string) => request<JobSnapshot>(`/api/jobs/${encodeURIComponent(id)}`),
  startRun: (body: RunRequest) =>
    request<JobStarted>('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  startCompare: (body: Omit<RunRequest, 'mode'>) =>
    request<JobStarted>('/api/runs/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
}

// URL gambar dan unduhan. Nama dataset dan run_id bisa memuat spasi atau huruf non-ASCII: selalu di-encode.

export const urls = {
  datasetPreview: (name: string, size = 480) => `/api/datasets/${encodeURIComponent(name)}/preview${query({ size })}`,
  runOriginal: (runId: string, stamp: string) => `/api/runs/${encodeURIComponent(runId)}/original${query({ v: stamp })}`,
  runMap: (runId: string, names: string[], view: MapView, stamp: string) =>
    `/api/runs/${encodeURIComponent(runId)}/map${query({ names: JSON.stringify(names), view, v: stamp })}`,
  runExport: (runId: string, kind: ExportKind, names: string[]) =>
    `/api/runs/${encodeURIComponent(runId)}/export/${kind}${query({ names: JSON.stringify(names) })}`,
}
