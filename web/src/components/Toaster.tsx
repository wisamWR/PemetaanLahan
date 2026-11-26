import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { CircleAlert, CircleCheck, Info, LoaderCircle, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export type ToastKind = 'success' | 'error' | 'info' | 'loading'

export interface ToastOptions {
  kind?: ToastKind
  title: string
  description?: string
  /** Milidetik sampai hilang sendiri; null = menetap sampai ditutup. Bawaan: success/info 5 dtk, error 9 dtk, loading menetap. */
  duration?: number | null
}

interface ToastItem extends ToastOptions {
  id: number
  kind: ToastKind
}

interface ToastApi {
  /** Menampilkan toast baru dan mengembalikan id-nya. */
  show: (options: ToastOptions) => number
  /** Mengubah toast yang ada (mis. dari "loading" ke "success"). Id yang sudah ditutup diabaikan. */
  update: (id: number, patch: Partial<ToastOptions>) => void
  dismiss: (id: number) => void
}

const ToastContext = createContext<ToastApi | null>(null)

const DEFAULT_DURATION: Record<ToastKind, number | null> = { success: 5000, info: 5000, error: 9000, loading: null }
const MAX_TOASTS = 4

/**
 * Pemberitahuan singkat di pojok kanan bawah (Fibery, Midday, Gamma): kartu kecil dengan ikon status, judul tebal,
 * keterangan redup, dan tombol tutup. Jenis `loading` memuat bilah tak tentu (pola kartu progres Obvious) untuk pekerjaan
 * yang butuh beberapa detik, mis. menyiapkan PNG resolusi penuh. Di layar sempit kartu memenuhi lebar dan duduk di atas
 * bilah aksi bawah halaman.
 */
export function ToastProvider({ children }: { children: ReactNode }) {
  // `itemsRef` adalah sumber kebenaran: show/update/dismiss mengubah daftar seketika (tanpa menunggu render), jadi
  // update yang datang tepat setelah show tidak hilang; `items` hanya cerminan untuk digambar.
  const [items, setItems] = useState<ToastItem[]>([])
  const itemsRef = useRef<ToastItem[]>([])
  const commit = useCallback((next: ToastItem[]) => {
    itemsRef.current = next
    setItems(next)
  }, [])
  const nextId = useRef(1)
  const timers = useRef(new Map<number, number>())

  const clearTimer = useCallback((id: number) => {
    const t = timers.current.get(id)
    if (t !== undefined) {
      window.clearTimeout(t)
      timers.current.delete(id)
    }
  }, [])

  const dismiss = useCallback(
    (id: number) => {
      clearTimer(id)
      commit(itemsRef.current.filter((t) => t.id !== id))
    },
    [clearTimer, commit],
  )

  const arm = useCallback(
    (id: number, kind: ToastKind, duration: number | null | undefined) => {
      clearTimer(id)
      const ms = duration === undefined ? DEFAULT_DURATION[kind] : duration
      if (ms !== null) timers.current.set(id, window.setTimeout(() => dismiss(id), ms))
    },
    [clearTimer, dismiss],
  )

  const show = useCallback(
    (options: ToastOptions) => {
      const id = nextId.current++
      const kind = options.kind ?? 'info'
      commit([...itemsRef.current, { ...options, id, kind }].slice(-MAX_TOASTS))
      arm(id, kind, options.duration)
      return id
    },
    [arm, commit],
  )

  const update = useCallback(
    (id: number, patch: Partial<ToastOptions>) => {
      const current = itemsRef.current.find((t) => t.id === id)
      if (!current) return                                   // sudah ditutup (pemakai menekan X): jangan dimunculkan lagi
      const kind = patch.kind ?? current.kind
      commit(itemsRef.current.map((t) => (t.id === id ? { ...t, ...patch, kind } : t)))
      arm(id, kind, patch.duration)                          // jenis baru, jadi hitung ulang jam hilang-sendiri
    },
    [arm, commit],
  )

  useEffect(() => {
    const map = timers.current
    return () => map.forEach((t) => window.clearTimeout(t))
  }, [])

  const api = useMemo<ToastApi>(() => ({ show, update, dismiss }), [show, update, dismiss])

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        className="pointer-events-none fixed inset-x-3 bottom-[5.25rem] z-50 flex flex-col gap-2 md:inset-x-auto md:right-5 md:bottom-5 md:w-96"
        data-testid="toaster"
      >
        {items.map((t) => (
          <ToastCard key={t.id} toast={t} onClose={() => dismiss(t.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

const ICON: Record<ToastKind, { Icon: typeof Info; className: string }> = {
  success: { Icon: CircleCheck, className: 'text-emerald-600 dark:text-emerald-400' },
  error: { Icon: CircleAlert, className: 'text-destructive' },
  info: { Icon: Info, className: 'text-muted-foreground' },
  loading: { Icon: LoaderCircle, className: 'animate-spin text-foreground' },
}

function ToastCard({ toast, onClose }: { toast: ToastItem; onClose: () => void }) {
  const { Icon, className } = ICON[toast.kind]
  return (
    <div
      role={toast.kind === 'error' ? 'alert' : 'status'}
      data-testid={`toast-${toast.kind}`}
      className={cn(
        'pointer-events-auto flex flex-col gap-2 rounded-xl border bg-card p-3 text-card-foreground shadow-lg ring-1 ring-foreground/5',
        toast.kind === 'error' && 'border-destructive/40',
      )}
    >
      <div className="flex items-start gap-3">
        <Icon className={cn('mt-0.5 size-4 shrink-0', className)} aria-hidden />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <p className="text-sm leading-snug font-medium">{toast.title}</p>
          {toast.description && <p className="text-xs leading-relaxed text-muted-foreground [overflow-wrap:anywhere]">{toast.description}</p>}
        </div>
        <Button variant="ghost" size="icon-xs" aria-label="Tutup pemberitahuan" onClick={onClose}>
          <X />
        </Button>
      </div>
      {toast.kind === 'loading' && (
        <div className="h-1 overflow-hidden rounded-full bg-muted" aria-hidden>
          <div className="h-full w-2/5 rounded-full bg-foreground/70" style={{ animation: 'lc-indeterminate 1.4s ease-in-out infinite' }} />
        </div>
      )}
    </div>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext)
  if (ctx === null) throw new Error('useToast harus dipakai di dalam <ToastProvider>')
  return ctx
}
