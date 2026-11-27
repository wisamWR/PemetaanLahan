import { KpiCell, KpiStrip } from '@/components/KpiStrip'
import type { Compare, Config, RunSummary } from '@/lib/api'
import { fmtInt } from '@/lib/format'

function contextLine(run: RunSummary | null): string | null {
  if (!run) return null
  return `${run.dataset} · K=${run.k} · ${run.iters} iterasi · seed ${run.seed} · ${fmtInt(run.width)}×${fmtInt(run.height)} (${fmtInt(run.n_pixels)} piksel)`
}

export function KpiRow({ data, config }: { data: Compare; config: Config }) {
  const { cpu, opt } = data
  const colors = config.device_colors
  const optAccent = opt ? (colors[opt.kmeans_device] ?? config.unknown_device_color) : colors.serial
  const speedupMissing = 'Butuh kedua run dengan dataset, K, iterasi, dan seed yang sama.'
  return (
    <KpiStrip label="Ringkasan kinerja" footer={contextLine(cpu ?? opt)}>
      <KpiCell
        testId="kpi-cpu"
        title={`Total: ${config.labels.modes.cpu}`}
        accent={colors.serial}
        value={cpu ? cpu.total_text : null}
        caption={cpu?.caption}
        extra={cpu?.note ?? undefined}
        empty="Belum dijalankan. Tekan Jalankan dengan mode CPU saja, atau Bandingkan."
        mock={cpu?.mock}
      />
      <KpiCell
        testId="kpi-opt"
        title={`Total: ${config.labels.modes.opt}`}
        accent={optAccent}
        value={opt ? opt.total_text : null}
        caption={opt?.caption}
        extra={opt?.note ?? undefined}
        empty="Belum dijalankan. Tekan Jalankan dengan mode Optimalisasi, atau Bandingkan."
        mock={opt?.mock}
      />
      <KpiCell
        testId="kpi-speedup"
        title="Speedup"
        accent="var(--foreground)"
        value={data.speedup != null ? data.speedup_text : null}
        caption="T_cpu / T_opt, dari total timing.json (bukan jumlah tahap)."
        empty={speedupMissing}
        mock={data.mock}
      />
    </KpiStrip>
  )
}
