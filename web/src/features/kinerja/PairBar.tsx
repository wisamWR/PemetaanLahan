import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { Compare, Group, PairSelection, RunSummary } from '@/lib/api'

const gid = (g: { dataset: string; k: number; iters: number; seed: number }) => `${g.dataset}|${g.k}|${g.iters}|${g.seed}`

function groupLabel(g: Group): string {
  return `${g.dataset} · K=${g.k} · ${g.iters} iterasi · seed ${g.seed}${g.complete ? '' : ' · belum lengkap'}`
}

function optLabel(r: RunSummary): string {
  const threads = r.kmeans_device === 'omp' && r.threads ? ` · ${r.threads} thread` : ''
  return `${r.engine_label}${threads} · ${r.created_text}`
}

/** Memilih pasangan sebanding (dataset, K, iterasi, seed sama) dan, bila ada beberapa, run Optimalisasinya. */
export function PairBar({
  groups,
  runs,
  data,
  onSelect,
}: {
  groups: Group[]
  runs: RunSummary[]
  data: Compare
  onSelect: (sel: PairSelection) => void
}) {
  const current = data.key ? gid(data.key) : undefined
  const group = groups.find((g) => gid(g) === current)
  const byId = new Map(runs.map((r) => [r.run_id, r]))
  const optRuns = (group?.opt_run_ids ?? []).map((id) => byId.get(id)).filter((r): r is RunSummary => r != null)

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Select
        value={current}
        onValueChange={(v) => {
          const g = groups.find((x) => gid(x) === v)
          if (g) onSelect({ key: { dataset: g.dataset, k: g.k, iters: g.iters, seed: g.seed } })
        }}
      >
        <SelectTrigger className="w-auto min-w-64 max-w-full" aria-label="Pasangan yang dibandingkan">
          <SelectValue placeholder="Pilih hasil" />
        </SelectTrigger>
        <SelectContent align="end">
          {groups.map((g) => (
            <SelectItem key={gid(g)} value={gid(g)}>
              {groupLabel(g)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {optRuns.length > 1 && data.opt && data.key && (
        <Select
          value={data.opt.run_id}
          onValueChange={(opt) => onSelect({ key: data.key!, opt })}
        >
          <SelectTrigger className="w-auto min-w-56 max-w-full" aria-label="Run Optimalisasi">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="end">
            {optRuns.map((r) => (
              <SelectItem key={r.run_id} value={r.run_id}>
                {optLabel(r)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
