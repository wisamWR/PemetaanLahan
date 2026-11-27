import { useState } from 'react'
import { MockBadge } from '@/components/MockBadge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableFooter, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { Chart, ChartRow } from '@/lib/api'
import { StageChart, type View } from './StageChart'

function Dot({ color }: { color: string }) {
  return <span className="inline-block size-2.5 shrink-0 rounded-sm" style={{ background: color }} aria-hidden />
}

function StageTable({ chart }: { chart: Chart }) {
  const hasSpeedup = chart.stage_speedup_text != null
  return (
    <Table data-testid="stage-table">
      <TableHeader>
        <TableRow>
          <TableHead>Tahap</TableHead>
          {chart.rows.map((r) => (
            <TableHead key={r.run_id} className="text-right">
              {r.label}
            </TableHead>
          ))}
          {hasSpeedup && <TableHead className="text-right">Speedup</TableHead>}
        </TableRow>
      </TableHeader>
      <TableBody>
        {chart.stages.map(({ stage, name }) => (
          <TableRow key={stage}>
            <TableCell className="font-medium">{name}</TableCell>
            {chart.rows.map((r: ChartRow) => {
              const c = r.stages.find((s) => s.stage === stage)!
              return (
                <TableCell key={r.run_id} className="text-right">
                  <div className="tabular-nums">{c.duration_text}</div>
                  <div className="flex items-center justify-end gap-1.5 text-xs text-muted-foreground">
                    <Dot color={c.color} />
                    {c.device_name} · {c.share_text}
                  </div>
                </TableCell>
              )
            })}
            {hasSpeedup && (
              <TableCell className="text-right tabular-nums text-muted-foreground">{chart.stage_speedup_text![stage]}</TableCell>
            )}
          </TableRow>
        ))}
      </TableBody>
      <TableFooter>
        <TableRow>
          <TableCell className="text-muted-foreground">Jumlah tahap</TableCell>
          {chart.rows.map((r) => (
            <TableCell key={r.run_id} className="text-right tabular-nums text-muted-foreground">
              {r.sum_text}
            </TableCell>
          ))}
          {hasSpeedup && <TableCell />}
        </TableRow>
        <TableRow>
          <TableCell className="font-semibold">{chart.total_name}</TableCell>
          {chart.rows.map((r) => (
            <TableCell key={r.run_id} className="text-right font-semibold tabular-nums">
              {r.total_text ?? '—'}
            </TableCell>
          ))}
          {hasSpeedup && <TableCell />}
        </TableRow>
      </TableFooter>
    </Table>
  )
}

export function StageChartCard({ chart }: { chart: Chart }) {
  const [view, setView] = useState<View>('seconds')
  return (
    <Card data-testid="stage-card">
      <CardHeader>
        {/* Judul + keterangan di kiri, kontrol tampilan di kanan header kartu (AirOps, Calendly, Gumloop). */}
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-3">
          <div className="flex min-w-0 flex-1 basis-64 flex-col gap-1">
            <CardTitle className="flex items-center gap-2">
              Waktu per tahap
              {chart.mock && <MockBadge />}
            </CardTitle>
            <CardDescription className="text-[13px]">
              {view === 'seconds'
                ? 'Waktu tiap tahap dalam detik. Garis tegak = total resmi dari timing.json, bukan jumlah segmen.'
                : 'Pangsa tiap tahap dari jumlah waktu tahap (bukan dari total resmi), supaya pergeseran bottleneck terlihat.'}
            </CardDescription>
          </div>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={view}
            onValueChange={(v) => v && setView(v as View)}
            aria-label="Tampilan grafik"
          >
            <ToggleGroupItem value="seconds">Detik</ToggleGroupItem>
            <ToggleGroupItem value="percent">Persentase</ToggleGroupItem>
          </ToggleGroup>
        </div>
      </CardHeader>
      <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:items-start xl:gap-x-0">
        <div className="min-w-0 xl:pr-6">
          <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-label="Legenda perangkat">
            {chart.devices.map((d) => (
              <li key={d.device} className="flex items-center gap-1.5">
                <Dot color={d.color} />
                {d.name}
              </li>
            ))}
          </ul>
          <StageChart chart={chart} view={view} />
          <p className="mt-2 text-xs text-muted-foreground">
            Tulisan yang tidak muat di segmen disembunyikan; semua nilai ada di hover dan tabel.
          </p>
        </div>
        <div className="min-w-0 overflow-x-auto xl:border-l xl:pl-6">
          <StageTable chart={chart} />
        </div>
      </CardContent>
    </Card>
  )
}
