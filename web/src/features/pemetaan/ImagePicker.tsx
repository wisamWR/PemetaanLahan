import { useRef, useState } from 'react'
import { Check, ImageIcon, LoaderCircle, TriangleAlert, Upload } from 'lucide-react'
import { MockBadge } from '@/components/MockBadge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { urls, type Config, type DatasetCard } from '@/lib/api'
import { cn } from '@/lib/utils'

/** Kartu bergambar satu citra: gambar di atas, nama dan ukuran di bawah; terpilih = garis tepi tegas + centang (Perplexity, Etsy). */
function DatasetTile({ d, selected, onSelect }: { d: DatasetCard; selected: boolean; onSelect: () => void }) {
  const [failed, setFailed] = useState(false)
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      aria-label={`${d.name}, ${d.size_text}`}
      data-testid={`dataset-${d.name}`}
      onClick={onSelect}
      className={cn(
        'flex min-w-0 flex-col gap-2 rounded-xl border bg-card p-2 text-left transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50',
        selected ? 'border-foreground ring-1 ring-foreground' : 'hover:border-foreground/30 hover:bg-accent/40',
      )}
    >
      <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-muted">
        {d.has_preview && !failed ? (
          <img
            src={urls.datasetPreview(d.name, 360)}
            alt=""
            loading="lazy"
            className="size-full object-cover"
            onError={() => setFailed(true)}
          />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-1 text-muted-foreground">
            <ImageIcon className="size-6" />
            <span className="text-[11px]">tanpa pratinjau</span>
          </div>
        )}
        {d.mock && <MockBadge className="absolute top-1.5 left-1.5 bg-background/85 backdrop-blur-sm" />}
        {selected && (
          <span className="absolute top-1.5 right-1.5 flex size-5 items-center justify-center rounded-full bg-foreground text-background">
            <Check className="size-3.5" />
          </span>
        )}
      </div>
      <div className="min-w-0 px-0.5 pb-0.5">
        <p className="truncate text-sm font-medium">{d.name}</p>
        <p className="truncate text-xs text-muted-foreground">{d.size_text}</p>
      </div>
    </button>
  )
}

/** Ubin unggah putus-putus di ujung grid (HoneyBook, AutoSend): klik atau seret berkas PNG ke sini. */
function UploadTile({ uploading, onFile }: { uploading: boolean; onFile: (file: File) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        if (!uploading) setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const file = e.dataTransfer.files[0]
        if (file && !uploading) onFile(file)
      }}
      className="contents"
    >
      <button
        type="button"
        disabled={uploading}
        data-testid="upload-tile"
        onClick={() => input.current?.click()}
        className={cn(
          'flex min-w-0 flex-col gap-2 rounded-xl border border-dashed bg-card/40 p-2 text-left transition-colors outline-none hover:border-foreground/40 hover:bg-accent/40 focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-wait',
          over && 'border-foreground bg-accent',
        )}
      >
        <div className="flex aspect-[4/3] flex-col items-center justify-center gap-1.5 rounded-lg bg-muted/60 text-muted-foreground">
          {uploading ? <LoaderCircle className="size-6 animate-spin" /> : <Upload className="size-6" />}
        </div>
        <div className="min-w-0 px-0.5 pb-0.5">
          <p className="truncate text-sm font-medium">{uploading ? 'Mengunggah…' : 'Unggah citra'}</p>
          <p className="truncate text-xs text-muted-foreground">
            <span className="sm:hidden">Pilih berkas PNG</span>
            <span className="hidden sm:inline">Seret PNG ke sini atau pilih berkas</span>
          </p>
        </div>
      </button>
      <input
        ref={input}
        type="file"
        accept="image/png"
        className="hidden"
        data-testid="upload-input"
        onChange={(e) => {
          const file = e.target.files?.[0]
          e.target.value = ''
          if (file) onFile(file)
        }}
      />
    </div>
  )
}

interface Props {
  config: Config
  datasets: DatasetCard[]
  uploadHint: string
  dataset: string
  k: number
  uploading: boolean
  uploadError: string | null
  onDataset: (name: string) => void
  onK: (k: number) => void
  onUpload: (file: File) => void
}

/** Kartu "Citra": grid kartu bergambar + ubin unggah, lalu parameter K. */
export function ImagePicker({ config, datasets, uploadHint, dataset, k, uploading, uploadError, onDataset, onK, onUpload }: Props) {
  const lim = config.limits
  return (
    <Card>
      <CardHeader>
        <CardTitle>Citra</CardTitle>
        <CardDescription className="text-[13px]">
          Pilih citra yang dipetakan, atau unggah PNG sendiri. {uploadHint}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div
          role="radiogroup"
          aria-label="Citra"
          className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5"
          data-testid="dataset-grid"
        >
          {datasets.map((d) => (
            <DatasetTile key={d.name} d={d} selected={d.name === dataset} onSelect={() => onDataset(d.name)} />
          ))}
          <UploadTile uploading={uploading} onFile={onUpload} />
        </div>

        {uploadError && (
          <Alert variant="destructive" data-testid="upload-error">
            <TriangleAlert />
            <AlertDescription>{uploadError}</AlertDescription>
          </Alert>
        )}

        <div className="flex flex-col gap-2 border-t pt-4 sm:flex-row sm:items-center sm:gap-6">
          <Label className="shrink-0 text-xs font-medium text-muted-foreground sm:w-40">Jumlah cluster (K)</Label>
          <div className="flex flex-1 items-center gap-3 sm:max-w-md">
            <Slider
              aria-label="Jumlah cluster (K)"
              min={lim.k_min}
              max={lim.k_max}
              step={1}
              value={[k]}
              onValueChange={([next]) => onK(next)}
            />
            <span className="w-6 text-right text-sm font-medium tabular-nums" data-testid="k-value">
              {k}
            </span>
          </div>
          <p className="text-xs text-muted-foreground">Tombol Buat peta memakai mode {config.labels.modes.opt}.</p>
        </div>
      </CardContent>
    </Card>
  )
}
