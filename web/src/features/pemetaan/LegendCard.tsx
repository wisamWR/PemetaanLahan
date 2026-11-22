import { TriangleAlert } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { AreaResponse } from '@/lib/api'
import type { ClusterEntry, Rows } from '@/lib/labelStore'
import { cn } from '@/lib/utils'

const NONE = '__none__'          // Radix Select tidak menerima nilai kosong untuk item
const BLANK: ClusterEntry = { kelas: null, nama_lain: '' }

interface Props {
  area: AreaResponse
  rows: Rows
  onChange: (cluster: number, patch: Partial<ClusterEntry>) => void
}

/**
 * Panel pelabelan sebagai legenda (Felt, Fresha, Midday): satu baris per cluster dengan swatch warna
 * peta, id cluster, persen, lalu Kelas (pilihan) dan Nama lain (teks bebas). Aturan "Nama lain
 * mengalahkan Kelas" dan penolakan nama `Total` diputuskan server; baris ini hanya menampilkan hasilnya.
 */
export function LegendCard({ area, rows, onChange }: Props) {
  return (
    <Card data-testid="legend-card" className="min-w-0">
      <CardHeader>
        <CardTitle>Legenda dan pelabelan</CardTitle>
        <CardDescription className="text-[13px]">{area.help.panel}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <ul className="-mx-1 flex max-h-[46rem] flex-col divide-y overflow-y-auto px-1" aria-label="Cluster">
          {area.clusters.map((c) => {
            const entry = rows[c.cluster] ?? BLANK
            const conflict = entry.kelas !== null && entry.nama_lain.trim() !== ''
            return (
              <li key={c.cluster} className="flex flex-col gap-2 py-3 first:pt-0 last:pb-0" data-testid={`legend-row-${c.cluster}`}>
                <div className="flex items-center gap-2.5">
                  <span
                    className="size-4 shrink-0 rounded-md ring-1 ring-foreground/15"
                    style={{ background: c.class_hex }}
                    title={`Warna di peta ${c.class_hex}; centroid ${c.centroid_hex}`}
                    data-testid={`swatch-${c.cluster}`}
                    aria-hidden
                  />
                  <span className="text-sm font-medium">Cluster {c.cluster}</span>
                  {conflict && <span className="min-w-0 truncate text-xs text-muted-foreground">dipakai: {c.name}</span>}
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums">{c.percent_text}</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Select
                    value={entry.kelas ?? ''}
                    onValueChange={(v) => onChange(c.cluster, { kelas: v === NONE ? null : v })}
                  >
                    <SelectTrigger size="sm" className="w-full" aria-label={`Kelas cluster ${c.cluster}`} title={area.help.kelas}>
                      <SelectValue placeholder="Kelas" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NONE}>Tanpa kelas</SelectItem>
                      {area.presets.map((p) => (
                        <SelectItem key={p} value={p}>
                          {p}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input
                    className={cn('h-8 text-sm', c.invalid && 'border-destructive')}
                    placeholder="Nama lain"
                    aria-label={`Nama lain cluster ${c.cluster}`}
                    aria-invalid={c.invalid || undefined}
                    title={area.help.other}
                    maxLength={area.name_max_chars}
                    value={entry.nama_lain}
                    onChange={(e) => onChange(c.cluster, { nama_lain: e.target.value })}
                  />
                </div>
              </li>
            )
          })}
        </ul>
        {area.errors.map((message) => (
          <Alert key={message} variant="destructive" data-testid="label-error">
            <TriangleAlert />
            <AlertDescription>{message}</AlertDescription>
          </Alert>
        ))}
        <p className="text-xs text-muted-foreground">{area.help.other}</p>
      </CardContent>
    </Card>
  )
}
