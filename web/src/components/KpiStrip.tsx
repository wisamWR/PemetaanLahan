import type { ReactNode } from 'react'
import { MockBadge } from '@/components/MockBadge'
import { Card } from '@/components/ui/card'
import { splitValue } from '@/lib/format'
import { cn } from '@/lib/utils'

export interface KpiCellProps {
  title: string
  /** Warna garis aksen tipis di atas sel (Ferndesk): warna perangkat di Kinerja, netral di Pemetaan. */
  accent: string
  value: string | null
  caption?: string
  extra?: string | null
  empty?: string
  mock?: boolean
  testId: string
  /** `auto`: nilai yang bukan angka (mis. nama kelas) memakai huruf lebih kecil supaya tidak terpotong. */
  valueSize?: 'lg' | 'auto'
  className?: string
}

/** Satu sel strip KPI: label kecil + angka besar + keterangan; garis aksen tipis di atas. */
export function KpiCell({ title, accent, value, caption, extra, empty, mock, testId, valueSize = 'lg', className }: KpiCellProps) {
  const [number, unit] = value ? splitValue(value) : ['—', '']
  const numeric = /^[\d.,]+$/.test(number)
  const small = valueSize === 'auto' && !numeric
  return (
    <div className={cn('relative flex min-w-0 flex-col gap-1.5 px-5 py-4', className)} data-testid={testId}>
      <div className="absolute inset-x-0 top-0 h-0.5" style={{ background: accent }} aria-hidden />
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <span className="text-[13px] text-muted-foreground">{title}</span>
        {mock && value && <MockBadge />}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span
          className={cn(
            'font-semibold tracking-tight tabular-nums',
            small ? 'min-w-0 text-xl leading-tight break-words' : 'text-3xl',
          )}
        >
          {number}
          {unit === '×' ? unit : ''}
        </span>
        {unit && unit !== '×' && <span className="text-base text-muted-foreground">{unit}</span>}
      </div>
      {value ? (
        <>
          {caption && <p className="text-xs text-muted-foreground">{caption}</p>}
          {extra && <p className="text-xs text-muted-foreground">{extra}</p>}
        </>
      ) : (
        <p className="text-xs text-muted-foreground">{empty}</p>
      )}
    </div>
  )
}

/**
 * Strip KPI bersambung: SATU kartu, sel dipisah garis tipis (Ferndesk, Mintlify, Vercel); bertumpuk di layar
 * sempit. `gridClassName` mengganti susunan bawaan (tiga sel) untuk halaman lain, mis. 2×2 atau 4 sel.
 */
export function KpiStrip({
  label,
  children,
  footer,
  gridClassName = 'grid divide-y md:grid-cols-3 md:divide-x md:divide-y-0',
}: {
  label: string
  children: ReactNode
  footer?: ReactNode
  gridClassName?: string
}) {
  return (
    <section aria-label={label}>
      <Card className="gap-0 py-0">
        <div className={gridClassName}>{children}</div>
        {footer && <p className="border-t bg-muted/40 px-5 py-2.5 text-xs text-muted-foreground">{footer}</p>}
      </Card>
    </section>
  )
}
