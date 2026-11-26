import { lazy, Suspense, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Gauge, Layers, LoaderCircle, Map as MapIcon } from 'lucide-react'
import { ThemeToggle } from '@/components/ThemeToggle'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'

// Halaman dimuat terpisah (code-splitting): Recharts (halaman Kinerja) tidak ikut unduhan awal halaman Pemetaan.
const KinerjaPage = lazy(() => import('@/features/kinerja/KinerjaPage').then((m) => ({ default: m.KinerjaPage })))
const PemetaanPage = lazy(() => import('@/features/pemetaan/PemetaanPage').then((m) => ({ default: m.PemetaanPage })))

function PageFallback() {
  return (
    <div className="flex items-center justify-center gap-2 py-24 text-sm text-muted-foreground" role="status">
      <LoaderCircle className="size-4 animate-spin" aria-hidden />
      Memuat halaman…
    </div>
  )
}

type Page = 'kinerja' | 'pemetaan'

const NAV: { id: Page; label: string; icon: typeof Gauge }[] = [
  { id: 'kinerja', label: 'Kinerja', icon: Gauge },
  { id: 'pemetaan', label: 'Pemetaan', icon: MapIcon },
]

const APP_NAME = 'Pemetaan Tutupan Lahan'

function pageFromHash(): Page {
  return window.location.hash === '#/pemetaan' ? 'pemetaan' : 'kinerja'
}

function usePage(): [Page, (p: Page) => void] {
  const [page, setPage] = useState<Page>(pageFromHash)
  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  return [page, (p) => (window.location.hash = `#/${p}`)]
}

/** Rel ikon di kiri (md ke atas): ikon dengan label kecil di bawahnya, pola rel Sentry dan Mintlify. */
function Rail({ page, onPage }: { page: Page; onPage: (p: Page) => void }) {
  return (
    <aside
      className="fixed inset-y-0 left-0 z-30 hidden w-16 flex-col items-center gap-1 border-r bg-card py-3 md:flex"
      aria-label="Navigasi utama"
    >
      <div className="mb-3 flex size-9 items-center justify-center rounded-lg bg-foreground text-background" aria-hidden>
        <Layers className="size-5" />
      </div>
      <nav className="flex flex-col items-center gap-1">
        {NAV.map(({ id, label, icon: Icon }) => {
          const active = page === id
          return (
            <button
              key={id}
              type="button"
              aria-current={active ? 'page' : undefined}
              onClick={() => onPage(id)}
              className={cn(
                'flex w-14 flex-col items-center gap-1 rounded-lg py-2 text-[10px] leading-none font-medium transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50',
                active ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
              )}
            >
              <Icon className="size-[18px]" />
              {label}
            </button>
          )
        })}
      </nav>
      <div className="mt-auto">
        <ThemeToggle />
      </div>
    </aside>
  )
}

export default function App() {
  const [page, setPage] = usePage()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config })
  const title = NAV.find((n) => n.id === page)!.label

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Rail page={page} onPage={setPage} />

      <div className="md:pl-16">
        {/* Bilah atas tipis: remah roti di kiri, status backend dan tema di kanan (Vapi, Mintlify). */}
        <header className="sticky top-0 z-20 flex h-12 items-center gap-3 border-b bg-background/85 px-4 backdrop-blur md:px-8">
          <div className="flex items-center gap-2 md:hidden">
            <Layers className="size-5" />
            {NAV.map(({ id, label }) => (
              <Button key={id} size="sm" variant={page === id ? 'secondary' : 'ghost'} onClick={() => setPage(id)}>
                {label}
              </Button>
            ))}
          </div>
          <nav aria-label="Remah roti" className="hidden min-w-0 items-center gap-1.5 text-sm md:flex">
            <span className="truncate text-muted-foreground">{APP_NAME}</span>
            <span className="text-muted-foreground/60" aria-hidden>
              /
            </span>
            <span className="font-medium">{title}</span>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            {config.data && (
              <Badge
                variant="outline"
                className={cn(
                  config.data.mock
                    ? 'border-amber-500/50 bg-amber-500/10 text-amber-700 dark:text-amber-300'
                    : 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
                )}
              >
                <span className="hidden sm:inline">{config.data.backend_label}</span>
                <span className="sm:hidden">{config.data.mock ? 'DATA TIRUAN' : 'core C++'}</span>
              </Badge>
            )}
            <div className="md:hidden">
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* pb-28 (layar sempit): ruang untuk bilah aksi tetap di dasar layar (lihat PageHeader). */}
        <main className="mx-auto w-full max-w-[1280px] px-4 pt-6 pb-28 md:px-8 md:py-8">
          <Suspense fallback={<PageFallback />}>{page === 'kinerja' ? <KinerjaPage /> : <PemetaanPage />}</Suspense>
        </main>
      </div>
    </div>
  )
}
