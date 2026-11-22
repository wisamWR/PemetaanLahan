import { CircleCheck, CircleX, Clock, LoaderCircle, SkipForward, X } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import type { JobSnapshot, JobStep } from '@/lib/api'
import type { Tick } from '@/lib/useJob'

function StepIcon({ status }: { status: JobStep['status'] }) {
  switch (status) {
    case 'running':
      return <LoaderCircle className="size-4 animate-spin text-foreground" aria-label="berjalan" />
    case 'done':
      return <CircleCheck className="size-4 text-emerald-600 dark:text-emerald-400" aria-label="selesai" />
    case 'error':
      return <CircleX className="size-4 text-destructive" aria-label="gagal" />
    case 'skipped':
      return <SkipForward className="size-4 text-muted-foreground" aria-label="dilewati" />
    default:
      return <Clock className="size-4 text-muted-foreground" aria-label="menunggu" />
  }
}

function stepText(step: JobStep, tick: Tick | null): string {
  if (step.status === 'running') {
    const t = tick && tick.step === step.index ? tick.step_elapsed_s : null
    return t == null ? 'berjalan…' : `berjalan: ${t.toFixed(1)} s`
  }
  if (step.status === 'done') return step.reused ? 'dipakai ulang (tidak dihitung ulang)' : 'selesai'
  if (step.status === 'error') return 'gagal'
  if (step.status === 'skipped') return 'dilewati'
  return 'menunggu'
}

/** Progres pekerjaan. Waktu berjalan hanya penanda progres; angka kinerja dari timing.json. */
export function JobPanel({ job, tick, onDismiss }: { job: JobSnapshot; tick: Tick | null; onDismiss: () => void }) {
  const running = job.status === 'running' || job.status === 'queued'
  return (
    <div className="flex flex-col gap-3" data-testid="job-panel" data-status={job.status}>
      <Card>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              {running ? <LoaderCircle className="size-4 animate-spin" /> : null}
              {running ? 'Run sedang berjalan' : job.status === 'done' ? 'Run selesai' : 'Run berhenti karena galat'}
              {running && tick && (
                <span className="font-normal tabular-nums text-muted-foreground">· {tick.elapsed_s.toFixed(1)} s</span>
              )}
            </div>
            {!running && (
              <Button variant="ghost" size="icon-xs" aria-label="Tutup" onClick={onDismiss}>
                <X />
              </Button>
            )}
          </div>

          <ol className="flex flex-col gap-2">
            {job.steps.map((s) => (
              <li key={s.index} className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                <StepIcon status={s.status} />
                <span className="font-medium">{s.label}</span>
                <span className="text-muted-foreground">{stepText(s, tick)}</span>
                {s.note && <span className="text-xs text-muted-foreground">· {s.note}</span>}
              </li>
            ))}
          </ol>

          {running && job.lines.length > 0 && (
            <pre className="max-h-32 overflow-auto rounded-lg bg-muted px-3 py-2 font-mono text-xs leading-relaxed text-muted-foreground">
              {job.lines.slice(-6).join('\n')}
            </pre>
          )}
          {running && (
            <p className="text-xs text-muted-foreground">
              Waktu berjalan di atas hanya penanda progres. Angka kinerja dibaca dari timing.json setelah run selesai.
            </p>
          )}
        </CardContent>
      </Card>

      {job.status === 'error' && job.error && (
        <Alert variant="destructive">
          <CircleX />
          <AlertTitle>{job.error.message}</AlertTitle>
          {job.error.detail && (
            <AlertDescription>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-xs">{job.error.detail}</pre>
            </AlertDescription>
          )}
        </Alert>
      )}
    </div>
  )
}
