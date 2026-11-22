import { ChevronDown, GitCompareArrows, LoaderCircle, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { Config, Dataset, KmeansChoice, Mode } from '@/lib/api'

export interface FormState {
  dataset: string
  k: number
  iters: string
  seed: string
  threads: number
  kmeans: KmeansChoice
  mode: Mode
  force: boolean
}

interface Props {
  config: Config
  datasets: Dataset[]
  form: FormState
  onChange: (patch: Partial<FormState>) => void
}

/**
 * Aksi utama halaman (Jalankan, Bandingkan). Diletakkan di kanan kepala halaman, seperti tombol
 * utama di pojok kanan atas hampir semua referensi dashboard (Sentry, Pinterest, Calendly, Framer).
 */
export function RunActions({
  config,
  mode,
  disabled,
  busy = false,
  onRun,
  onCompare,
}: {
  config: Config
  mode: Mode
  disabled: boolean
  /** Ada run yang sedang berjalan: tombol utama menampilkan "Berjalan…" dengan pemutar, bukan sekadar redup. */
  busy?: boolean
  onRun: () => void
  onCompare: () => void
}) {
  return (
    <>
      <Button variant="outline" disabled={disabled} onClick={onRun}>
        <Play />
        Jalankan<span className="hidden sm:inline"> {config.labels.modes[mode]}</span>
      </Button>
      <Button disabled={disabled} onClick={onCompare} aria-busy={busy || undefined}>
        {busy ? <LoaderCircle className="animate-spin" /> : <GitCompareArrows />}
        {busy ? (
          'Berjalan…'
        ) : (
          <>
            Bandingkan<span className="hidden sm:inline"> (CPU saja lalu Optimalisasi)</span>
          </>
        )}
      </Button>
    </>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <Label className="text-xs font-medium text-muted-foreground">{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

export function ParamsCard({ config, datasets, form, onChange }: Props) {
  const lim = config.limits
  const kmeansChoices = config.core?.kmeans_choices ?? (['auto', 'omp', 'cuda'] as KmeansChoice[])
  const noDatasets = datasets.length === 0

  return (
    <Card size="sm">
      <CardContent className="flex flex-col gap-4">
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-[1.3fr_1.2fr_0.7fr_1.2fr]">
          <Field label="Dataset">
            <Select value={form.dataset} onValueChange={(dataset) => onChange({ dataset })} disabled={noDatasets}>
              <SelectTrigger className="w-full" aria-label="Dataset">
                <SelectValue placeholder={noDatasets ? 'Tidak ada dataset' : 'Pilih dataset'} />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((d) => (
                  <SelectItem key={d.name} value={d.name}>
                    {d.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field label="Jumlah cluster (K)">
            <div className="flex h-9 items-center gap-3">
              <Slider
                aria-label="Jumlah cluster (K)"
                min={lim.k_min}
                max={lim.k_max}
                step={1}
                value={[form.k]}
                onValueChange={([k]) => onChange({ k })}
              />
              <span className="w-6 text-right text-sm font-medium tabular-nums">{form.k}</span>
            </div>
          </Field>

          <Field label="Iterasi">
            <Input
              type="number"
              inputMode="numeric"
              min={1}
              max={lim.iters_max}
              aria-label="Iterasi"
              value={form.iters}
              onChange={(e) => onChange({ iters: e.target.value })}
            />
          </Field>

          <Field label="Mode untuk tombol Jalankan">
            <ToggleGroup
              type="single"
              variant="outline"
              value={form.mode}
              onValueChange={(v) => v && onChange({ mode: v as Mode })}
              className="w-full"
              aria-label="Mode"
            >
              <ToggleGroupItem value="cpu" className="flex-1">
                {config.labels.modes.cpu}
              </ToggleGroupItem>
              <ToggleGroupItem value="opt" className="flex-1">
                {config.labels.modes.opt}
              </ToggleGroupItem>
            </ToggleGroup>
          </Field>
        </div>

        {config.core?.notice && (
          <p className="rounded-lg border border-dashed px-3 py-2 text-xs text-muted-foreground">{config.core.notice}</p>
        )}

        <Collapsible className="flex flex-col gap-3">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="group -ml-2 self-start text-muted-foreground">
              Lanjutan
              <ChevronDown className="transition-transform group-data-[state=open]:rotate-180" />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="Seed inisialisasi">
                <Input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={lim.seed_max}
                  aria-label="Seed inisialisasi"
                  value={form.seed}
                  onChange={(e) => onChange({ seed: e.target.value })}
                />
              </Field>
              <Field label="K-Means (mode Optimalisasi)">
                <Select value={form.kmeans} onValueChange={(v) => onChange({ kmeans: v as KmeansChoice })}>
                  <SelectTrigger className="w-full" aria-label="K-Means">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {kmeansChoices.map((c) => (
                      <SelectItem key={c} value={c}>
                        {config.labels.kmeans[c] ?? c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Thread OpenMP">
                <div className="flex h-9 items-center gap-3">
                  <Slider
                    aria-label="Thread OpenMP"
                    min={lim.threads_min}
                    max={lim.threads_max}
                    step={1}
                    value={[form.threads]}
                    onValueChange={([threads]) => onChange({ threads })}
                  />
                  <span className="w-6 text-right text-sm font-medium tabular-nums">{form.threads}</span>
                </div>
              </Field>
              <Field label="Paksa jalankan ulang" hint="Abaikan hasil CPU saja yang tersimpan di runs/.">
                <div className="flex h-9 items-center">
                  <Switch
                    aria-label="Paksa jalankan ulang"
                    checked={form.force}
                    onCheckedChange={(force) => onChange({ force })}
                  />
                </div>
              </Field>
            </div>

          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  )
}
