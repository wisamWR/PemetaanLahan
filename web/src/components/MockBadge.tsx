import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

/** Lencana wajib untuk hasil tiruan (kartu, grafik, verifikasi). Teks tetap "DATA TIRUAN". */
export function MockBadge({ className }: { className?: string }) {
  return (
    <Badge
      variant="outline"
      className={cn(
        'border-amber-500/50 bg-amber-500/10 px-1.5 text-[11px] font-semibold tracking-wide text-amber-700 dark:text-amber-300',
        className,
      )}
    >
      DATA TIRUAN
    </Badge>
  )
}
