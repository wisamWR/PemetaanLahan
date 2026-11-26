import { useState, type ReactNode } from 'react'
import { ImageOff } from 'lucide-react'
import { MockBadge } from '@/components/MockBadge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { urls, type AreaResponse, type MapView } from '@/lib/api'
import { CompareSlider, CornerPill } from './CompareSlider'
import { useImage } from './hooks'

type Mode = 'slider' | 'side'

function Missing({ children }: { children: ReactNode }) {
  return (
    <div className="flex size-full flex-col items-center justify-center gap-2 bg-muted px-6 text-center text-sm text-muted-foreground">
      <ImageOff className="size-6" />
      <p className="max-w-xs">{children}</p>
    </div>
  )
}

/**
 * Citra asli | peta kelas. Slider sebelum/sesudah (satu garis, gagang bulat; pil kecil di sudut) atau
 * berdampingan (pengalih "View type" Customer.io). Peta dibuat server dari nama kelas (`/api/runs/{id}/map`),
 * jadi warna di peta sama dengan legenda dan ekspor; gambar baru menggantikan yang lama hanya setelah siap.
 */
export function MapCard({ area }: { area: AreaResponse }) {
  const [mode, setMode] = useState<Mode>('slider')
  const [view, setView] = useState<MapView>('class')
  const original = useImage(urls.runOriginal(area.run_id, area.stamp))
  const map = useImage(urls.runMap(area.run_id, area.names, view, area.stamp))
  const ratio = area.preview.width / area.preview.height

  const before =
    original.status === 'error' ? (
      <Missing>{original.reason}</Missing>
    ) : original.src ? (
      <img src={original.src} alt="Citra asli" draggable={false} className="size-full object-fill" />
    ) : (
      <Skeleton className="size-full rounded-none" />
    )
  const after =
    map.status === 'error' && !map.src ? (
      <Missing>{map.reason}</Missing>
    ) : map.src ? (
      <img
        src={map.src}
        alt="Peta kelas hasil pengelompokan"
        draggable={false}
        className="size-full object-fill [image-rendering:pixelated]"
        data-testid="class-map-image"
      />
    ) : (
      <Skeleton className="size-full rounded-none" />
    )
  const beforeLabel = 'Citra asli'
  const afterLabel = view === 'class' ? 'Peta kelas' : 'Warna centroid'

  return (
    <Card data-testid="map-card">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-3">
          <div className="flex min-w-0 flex-1 basis-56 flex-col gap-1">
            <CardTitle className="flex items-center gap-2">
              Citra asli dan peta kelas
              {area.mock && <MockBadge />}
            </CardTitle>
            <CardDescription className="text-[13px]">
              {mode === 'slider' ? 'Geser garis untuk membandingkan citra asli dengan peta kelas.' : 'Citra asli dan peta kelas berdampingan.'}
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={mode}
              onValueChange={(v) => v && setMode(v as Mode)}
              aria-label="Tampilan perbandingan"
            >
              <ToggleGroupItem value="slider">Geser</ToggleGroupItem>
              <ToggleGroupItem value="side">Berdampingan</ToggleGroupItem>
            </ToggleGroup>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={view}
              onValueChange={(v) => v && setView(v as MapView)}
              aria-label="Warna peta"
            >
              <ToggleGroupItem value="class">Warna kelas</ToggleGroupItem>
              <ToggleGroupItem value="centroid">Warna centroid</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {mode === 'slider' ? (
          <CompareSlider
            before={before}
            after={after}
            ratio={ratio}
            beforeLabel={beforeLabel}
            afterLabel={afterLabel}
            label="Geser untuk membandingkan citra asli dan peta kelas"
          />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              { node: before, label: beforeLabel },
              { node: after, label: afterLabel },
            ].map(({ node, label }) => (
              <div
                key={label}
                className="relative overflow-hidden rounded-lg bg-muted"
                style={{ aspectRatio: String(ratio) }}
              >
                {node}
                <CornerPill side="left">{label}</CornerPill>
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-muted-foreground">{area.preview_caption}</p>
      </CardContent>
    </Card>
  )
}
