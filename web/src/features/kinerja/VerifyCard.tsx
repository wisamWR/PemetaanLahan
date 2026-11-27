import { CircleCheck, CircleX, Info, TriangleAlert } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { MockBadge } from '@/components/MockBadge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { api, type Compare, type Level, type Verification } from '@/lib/api'
import { cn } from '@/lib/utils'

const TITLE = 'Status verifikasi antarmode'

const LEVEL_STYLE: Record<Level, { icon: typeof Info; box: string }> = {
  ok: { icon: CircleCheck, box: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' },
  partial: { icon: Info, box: 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300' },
  bad: { icon: CircleX, box: 'border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300' },
  warn: { icon: TriangleAlert, box: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300' },
}

function Verdict({ v }: { v: Verification }) {
  const { icon: Icon, box } = LEVEL_STYLE[v.level]
  return (
    <div className="flex flex-col gap-4" data-testid="verify-result" data-level={v.level} data-status={v.status}>
      <div className={cn('flex items-start gap-3 rounded-lg border px-4 py-3', box)}>
        <Icon className="mt-0.5 size-5 shrink-0" aria-hidden />
        <p className="text-sm font-medium leading-relaxed">{v.verdict}</p>
      </div>
      {v.pair_label && <p className="text-xs text-muted-foreground">Pembanding: {v.pair_label}</p>}
      {v.source_text && <p className="text-xs text-muted-foreground">{v.source_text}</p>}
      {v.notes.map((n) => (
        <p key={n} className="text-xs text-muted-foreground">
          Catatan: {n}
        </p>
      ))}
      {v.mock_notice && <p className="text-xs font-medium text-amber-700 dark:text-amber-300">{v.mock_notice}</p>}
      {v.rows.length > 0 && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 border-t pt-3 text-sm">
          {v.rows.map((r) => (
            <div key={r.what} className="contents">
              <dt className="text-muted-foreground">{r.what}</dt>
              <dd>{r.result}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

export function VerifyCard({ data, version }: { data: Compare; version: string }) {
  const cpu = data.cpu
  const opt = data.opt
  const ready = cpu != null && opt != null
  const query = useQuery({
    queryKey: ['verify', cpu?.run_id, opt?.run_id, version],
    queryFn: () => api.verify(cpu!.run_id, opt!.run_id),
    enabled: ready,
  })
  const mock = data.mock

  return (
    <Card data-testid="verify-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {TITLE}
          {mock && <MockBadge />}
        </CardTitle>
        <CardDescription>
          Label dan centroid CPU saja dibandingkan dengan Optimalisasi. Hijau hanya bila label resolusi penuh identik.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!ready ? (
          <p className="text-sm text-muted-foreground">
            Butuh kedua run untuk membandingkan. Jalankan {cpu ? 'Optimalisasi' : 'CPU saja'} untuk{' '}
            {(cpu ?? opt)?.dataset} · K={(cpu ?? opt)?.k} · {(cpu ?? opt)?.iters} iterasi · seed {(cpu ?? opt)?.seed} (dataset, K,
            iterasi, dan seed harus sama).
          </p>
        ) : query.isPending ? (
          <Skeleton className="h-16 w-full" />
        ) : query.isError ? (
          <Alert variant="destructive">
            <AlertDescription>{query.error.message}</AlertDescription>
          </Alert>
        ) : (
          <Verdict v={query.data} />
        )}
      </CardContent>
    </Card>
  )
}
