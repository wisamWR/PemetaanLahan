import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useXAxisScale, useYAxisScale } from 'recharts'
import type { Chart, ChartRow, StageCell } from '@/lib/api'

export type View = 'seconds' | 'percent'

/** Layar sempit (< 640 px): sumbu-y dan ruang teks total dipersempit supaya area batang tetap terbaca. */
function useCompact(): boolean {
  const query = "(max-width: 639px)"
  const [compact, setCompact] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const on = () => setCompact(mq.matches)
    mq.addEventListener("change", on)
    return () => mq.removeEventListener("change", on)
  }, [])
  return compact
}

const ROW_PX = 76
const CHROME_PX = 58
const PERCENT_TICKS = [0, 25, 50, 75, 100]
const AXIS_TEXT = { fill: 'var(--muted-foreground)', fontSize: 12 }

function cellOf(row: ChartRow, stage: string): StageCell {
  return row.stages.find((s) => s.stage === stage)!
}

function tickSeconds(v: number): string {
  return `${Number(v.toFixed(v < 10 ? 1 : 0))} s`
}

/** Penanda total resmi (timing.json): garis tegak di x = total (tampilan detik) dan teksnya di ujung kanan. */
function TotalMarkers({ chart, view }: { chart: Chart; view: View }) {
  const xScale = useXAxisScale()
  const yScale = useYAxisScale()
  if (!xScale || !yScale) return null
  return (
    <g aria-hidden>
      {chart.rows.map((row) => {
        const yMid = yScale(row.label, { position: 'middle' })
        if (yMid == null || row.total_text == null) return null
        const x = view === 'seconds' ? xScale(row.total_s ?? 0) : xScale(100)
        if (x == null) return null
        return (
          <g key={row.run_id}>
            {view === 'seconds' && (
              <line x1={x} x2={x} y1={yMid - 20} y2={yMid + 20} stroke="var(--foreground)" strokeWidth={2.5} strokeLinecap="round" />
            )}
            <text x={x + 10} y={yMid} dominantBaseline="middle" fill="var(--foreground)" fontSize={13} fontWeight={500}>
              total {row.total_text}
            </text>
          </g>
        )
      })}
    </g>
  )
}

interface LabelProps {
  x?: number | string
  y?: number | string
  width?: number | string
  height?: number | string
  index?: number
}

function segmentLabel(chart: Chart, view: View, stage: string) {
  return function SegmentLabel(props: LabelProps) {
    const { x = 0, y = 0, width = 0, height = 0, index = 0 } = props
    const cell = chart.rows[index] ? cellOf(chart.rows[index], stage) : null
    if (!cell || cell.seconds <= 0) return null
    const text = view === 'seconds' ? cell.duration_text : cell.share_text
    const w = Number(width)
    if (w < text.length * 7.2 + 12) return null // tidak muat: nilai tetap ada di hover dan tabel
    return (
      <text
        x={Number(x) + w / 2}
        y={Number(y) + Number(height) / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#ffffff"
        fontSize={12}
        fontWeight={500}
      >
        {text}
      </text>
    )
  }
}

interface TooltipPayload {
  dataKey?: string | number
  payload?: { label?: string }
}

function ChartTooltip({ active, payload, chart, view }: { active?: boolean; payload?: TooltipPayload[]; chart: Chart; view: View }) {
  if (!active || !payload?.length) return null
  const stage = String(payload[0].dataKey)
  const row = chart.rows.find((r) => r.label === payload[0].payload?.label)
  const cell = row ? cellOf(row, stage) : null
  if (!row || !cell) return null
  return (
    <div className="min-w-48 rounded-lg border bg-popover px-3 py-2.5 text-sm text-popover-foreground shadow-md">
      <div className="mb-1.5 font-semibold">{row.label}</div>
      <div className="flex items-center gap-2">
        <span className="size-2.5 rounded-sm" style={{ background: cell.color }} />
        <span className="font-medium">{cell.name}</span>
        <span className="ml-auto tabular-nums">{view === 'seconds' ? cell.duration_text : cell.share_text}</span>
      </div>
      <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
        <dt>Perangkat</dt>
        <dd className="text-right text-foreground">{cell.device_name}</dd>
        <dt>Waktu</dt>
        <dd className="text-right tabular-nums text-foreground">{cell.duration_text}</dd>
        <dt>Pangsa</dt>
        <dd className="text-right tabular-nums text-foreground">{cell.share.toFixed(1)}% dari jumlah tahap</dd>
        {cell.detail.map((d) => (
          <div key={d.key} className="contents">
            <dt>{d.key}</dt>
            <dd className="text-right tabular-nums text-foreground">{d.text}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

export function StageChart({ chart, view }: { chart: Chart; view: View }) {
  const compact = useCompact()
  const data = chart.rows.map((row) => ({
    label: row.label,
    ...Object.fromEntries(row.stages.map((s) => [s.stage, view === 'seconds' ? s.seconds : s.share])),
  }))
  const upper = view === 'seconds' ? Math.max(chart.max_seconds, 1e-9) : 100
  const height = chart.rows.length * (compact ? 84 : ROW_PX) + CHROME_PX

  return (
    <div role="img" aria-label="Grafik waktu per tahap" data-testid="stage-chart" data-view={view}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: compact ? 84 : 116, bottom: 4, left: 0 }} barCategoryGap="40%">
          <CartesianGrid horizontal={false} stroke="var(--border)" />
          <XAxis
            type="number"
            domain={[0, upper]}
            ticks={view === 'percent' ? PERCENT_TICKS : undefined}
            tickFormatter={(v: number) => (view === 'percent' ? `${v}%` : tickSeconds(v))}
            tick={AXIS_TEXT}
            stroke="var(--border)"
            tickLine={false}
          />
          <YAxis type="category" dataKey="label" width={compact ? 88 : 150} tick={AXIS_TEXT} tickLine={false} stroke="var(--border)" />
          <Tooltip shared={false} cursor={false} content={<ChartTooltip chart={chart} view={view} />} isAnimationActive={false} />
          {chart.stages.map(({ stage }) => (
            <Bar key={stage} dataKey={stage} stackId="waktu" isAnimationActive={false} stroke="var(--card)" strokeWidth={2}>
              {chart.rows.map((row) => (
                <Cell key={row.run_id} fill={cellOf(row, stage).color} />
              ))}
              <LabelList dataKey={stage} content={segmentLabel(chart, view, stage)} />
            </Bar>
          ))}
          <TotalMarkers chart={chart} view={view} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
