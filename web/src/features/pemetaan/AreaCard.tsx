import { useRef } from 'react'
import { ChevronDown, Download, FileSpreadsheet, ImageIcon, Map as MapIcon } from 'lucide-react'
import { MockBadge } from '@/components/MockBadge'
import { useToast } from '@/components/Toaster'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { urls, type AreaResponse, type AreaRow, type ExportItem, type ExportKind } from '@/lib/api'
import { downloadFile } from '@/lib/download'
import { cn } from '@/lib/utils'

const EXPORT_ICONS: Record<ExportKind, typeof Download> = {
  map: MapIcon,
  area: FileSpreadsheet,
  clusters: FileSpreadsheet,
  full: ImageIcon,
}

function ExportEntry({ item, href, onPrepare }: { item: ExportItem; href: string; onPrepare?: () => void }) {
  const Icon = EXPORT_ICONS[item.kind]
  const body = (
    <>
      <Icon className="mt-0.5 text-muted-foreground" />
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="font-medium">{item.label}</span>
        {item.available ? (
          <span className="truncate text-xs text-muted-foreground" title={item.filename}>
            {item.hint ?? item.filename}
          </span>
        ) : (
          <span className="text-xs whitespace-normal text-muted-foreground">{item.reason}</span>
        )}
      </span>
    </>
  )
  if (!item.available) {
    return (
      <DropdownMenuItem disabled data-testid={`export-${item.kind}`}>
        {body}
      </DropdownMenuItem>
    )
  }
  if (onPrepare) {
    // Berkas yang butuh disiapkan server dulu: tidak lewat tautan, supaya penyiapannya diberi umpan balik (toast).
    return (
      <DropdownMenuItem data-testid={`export-${item.kind}`} onSelect={onPrepare}>
        {body}
      </DropdownMenuItem>
    )
  }
  return (
    <DropdownMenuItem asChild data-testid={`export-${item.kind}`}>
      <a href={href} download={item.filename}>
        {body}
      </a>
    </DropdownMenuItem>
  )
}

/**
 * Tombol "Ekspor" outline di kanan header kartu (HubSpot, Kajabi) membuka menu unduhan (Sketch, Adobe Express).
 * PNG resolusi penuh dibuat server dari `labels.u8` (±3 detik untuk tile 10980²), jadi diunduh lewat fetch dengan toast
 * "menyiapkan" lalu "siap" atau galatnya (pola kartu progres Obvious, Bonsai, Optimal Workshop); ekspor kecil tetap tautan.
 */
function ExportMenu({ area }: { area: AreaResponse }) {
  const toast = useToast()
  const preparing = useRef(false)

  const downloadFull = async (item: ExportItem, href: string) => {
    if (preparing.current) {
      toast.show({ kind: 'info', title: 'PNG resolusi penuh masih disiapkan', description: 'Tunggu sampai selesai sebelum meminta lagi.' })
      return
    }
    preparing.current = true
    const id = toast.show({
      kind: 'loading',
      title: 'Menyiapkan PNG resolusi penuh…',
      description: 'Citra besar butuh beberapa detik. Unduhan dimulai otomatis.',
    })
    try {
      const name = await downloadFile(href, item.filename)
      toast.update(id, { kind: 'success', title: 'PNG resolusi penuh siap', description: name })
    } catch (e) {
      toast.update(id, {
        kind: 'error',
        title: 'PNG resolusi penuh gagal dibuat',
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      preparing.current = false
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" data-testid="export-button">
          <Download />
          Ekspor
          <ChevronDown className="text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-80">
        <DropdownMenuLabel>Unduh hasil pemetaan</DropdownMenuLabel>
        {area.exports.map((item) => {
          const href = urls.runExport(area.run_id, item.kind, area.names)
          return (
            <ExportEntry
              key={item.kind}
              item={item}
              href={href}
              onPrepare={item.kind === 'full' ? () => void downloadFull(item, href) : undefined}
            />
          )
        })}
        {area.export_note && <DropdownMenuLabel className="pt-2">{area.export_note}</DropdownMenuLabel>}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function Swatches({ colors }: { colors: string[] }) {
  if (colors.length === 0) return <span className="inline-block size-3.5 shrink-0" aria-hidden />
  return (
    <span className="flex shrink-0 items-center" aria-hidden>
      {colors.slice(0, 4).map((c, i) => (
        <span key={`${c}-${i}`} className="-ml-1 size-3.5 rounded-sm ring-2 ring-card first:ml-0" style={{ background: c }} />
      ))}
    </span>
  )
}

/** Bilah persen tipis di sel tabel (Kajabi): lebar = persen dari seluruh piksel, warna = warna kelas. */
function ShareBar({ row }: { row: AreaRow }) {
  if (row.is_total) return null
  const color = row.unnamed ? 'var(--muted-foreground)' : (row.colors[0] ?? 'var(--muted-foreground)')
  return (
    <span className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-muted sm:block" aria-hidden>
      <span className="block h-full rounded-full" style={{ width: `${Math.min(100, Math.max(0, row.percent))}%`, background: color }} />
    </span>
  )
}

/** Tabel luas per kelas: persen dan hektare dari `counts` resolusi penuh (angka dan teks dari server). */
export function AreaCard({ area }: { area: AreaResponse }) {
  return (
    <Card data-testid="area-card">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-3">
          <div className="flex min-w-0 flex-1 basis-64 flex-col gap-1">
            <CardTitle className="flex items-center gap-2">
              Luas per kelas
              {area.mock && <MockBadge />}
            </CardTitle>
            <CardDescription className="text-[13px]">{area.area_caption}</CardDescription>
          </div>
          <ExportMenu area={area} />
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {/* Layar sempit (< sm): daftar bertumpuk, bukan tabel yang terpotong dan perlu digeser ke samping. */}
        <ul className="divide-y rounded-lg border sm:hidden" data-testid="area-list" aria-label="Luas per kelas">
          {area.area.map((row) => (
            <li
              key={row.is_total ? '__total' : row.name}
              className={cn('flex flex-col gap-1.5 px-3 py-2.5', row.is_total && 'bg-muted/40 font-semibold')}
            >
              <div className="flex items-center gap-2">
                {!row.is_total && <Swatches colors={row.colors} />}
                <span className={cn('min-w-0 flex-1 break-words', row.unnamed && 'text-muted-foreground italic')}>{row.name}</span>
                <span className="shrink-0 tabular-nums">{row.percent_text}</span>
              </div>
              {!row.is_total && (
                <span className="block h-1.5 overflow-hidden rounded-full bg-muted" aria-hidden>
                  <span
                    className="block h-full rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, row.percent))}%`, background: row.unnamed ? 'var(--muted-foreground)' : (row.colors[0] ?? 'var(--muted-foreground)') }}
                  />
                </span>
              )}
              <div className="flex flex-wrap items-baseline justify-between gap-x-3 text-xs font-normal text-muted-foreground">
                <span>{row.clusters ? `Cluster ${row.clusters}` : ''}</span>
                <span className="tabular-nums">{row.area_text}</span>
              </div>
            </li>
          ))}
        </ul>
        <Table data-testid="area-table" className="hidden sm:table">
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs text-muted-foreground">Kelas</TableHead>
              <TableHead className="hidden text-xs text-muted-foreground sm:table-cell">Cluster</TableHead>
              <TableHead className="hidden text-right text-xs text-muted-foreground md:table-cell">Piksel</TableHead>
              <TableHead className="text-right text-xs text-muted-foreground">Persen</TableHead>
              <TableHead className="text-right text-xs text-muted-foreground">Luas (ha)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {area.area.map((row) => (
              <TableRow
                key={row.is_total ? '__total' : row.name}
                className={cn(row.is_total && 'bg-muted/40 font-semibold hover:bg-muted/40')}
                data-testid={row.is_total ? 'area-total' : `area-row-${row.name}`}
              >
                <TableCell>
                  <span className="flex items-center gap-2">
                    {!row.is_total && <Swatches colors={row.colors} />}
                    <span className={cn('min-w-0 break-words', row.unnamed && 'text-muted-foreground italic')}>{row.name}</span>
                  </span>
                </TableCell>
                <TableCell className="hidden text-muted-foreground tabular-nums sm:table-cell">{row.clusters}</TableCell>
                <TableCell className="hidden text-right tabular-nums md:table-cell">{row.pixels_text}</TableCell>
                <TableCell className="text-right">
                  <span className="flex items-center justify-end gap-2.5">
                    <ShareBar row={row} />
                    <span className="tabular-nums">{row.percent_text}</span>
                  </span>
                </TableCell>
                <TableCell className="text-right tabular-nums">{row.area_text}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {area.mock_notice && <p className="text-xs text-muted-foreground">{area.mock_notice}</p>}
      </CardContent>
    </Card>
  )
}
