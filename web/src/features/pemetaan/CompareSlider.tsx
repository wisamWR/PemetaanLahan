import { useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { ChevronsLeftRight } from 'lucide-react'

const clamp = (n: number) => Math.min(100, Math.max(0, n))

/** Penanda "Sebelum/Sesudah" kecil di sudut gambar (ElevenLabs, Krea, Magnific): pil gelap tembus pandang. */
export function CornerPill({ children, side }: { children: ReactNode; side: 'left' | 'right' }) {
  return (
    <span
      className={`pointer-events-none absolute top-3 z-10 rounded-md bg-black/55 px-2 py-1 text-[11px] leading-none font-medium text-white backdrop-blur-sm ${
        side === 'left' ? 'left-3' : 'right-3'
      }`}
    >
      {children}
    </span>
  )
}

/**
 * Slider sebelum/sesudah buatan sendiri (tanpa paket): satu garis tipis dengan gagang bulat berpanah
 * (ElevenLabs, Krea, Leonardo). `before` di kiri garis, `after` di kanan. Kedua sisi memenuhi kotak yang SAMA
 * (`aspect-ratio` dari ukuran pratinjau), jadi piksel keduanya sejajar walau ukuran berkasnya berbeda.
 * Bisa digeser dengan penunjuk (mouse, sentuh) dan dengan papan ketik (panah, Home, End).
 */
export function CompareSlider({
  before,
  after,
  ratio,
  beforeLabel,
  afterLabel,
  label,
}: {
  before: ReactNode
  after: ReactNode
  ratio: number
  beforeLabel: string
  afterLabel: string
  label: string
}) {
  const box = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const [pos, setPos] = useState(50)

  const moveTo = (e: PointerEvent) => {
    const rect = box.current?.getBoundingClientRect()
    if (rect && rect.width > 0) setPos(clamp(((e.clientX - rect.left) / rect.width) * 100))
  }
  const onKeyDown = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 2
    const next =
      e.key === 'ArrowLeft' || e.key === 'ArrowDown'
        ? pos - step
        : e.key === 'ArrowRight' || e.key === 'ArrowUp'
          ? pos + step
          : e.key === 'Home'
            ? 0
            : e.key === 'End'
              ? 100
              : null
    if (next === null) return
    e.preventDefault()
    setPos(clamp(next))
  }

  return (
    <div
      ref={box}
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-orientation="horizontal"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      aria-valuetext={`${Math.round(pos)} % ${beforeLabel}, sisanya ${afterLabel}`}
      data-testid="compare-slider"
      className="relative mx-auto w-full cursor-ew-resize touch-pan-y overflow-hidden rounded-lg bg-muted outline-none select-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
      style={{ aspectRatio: String(ratio), maxWidth: `min(100%, calc(72vh * ${ratio}))` }}
      onPointerDown={(e) => {
        dragging.current = true
        e.currentTarget.setPointerCapture(e.pointerId)
        moveTo(e)
      }}
      onPointerMove={(e) => dragging.current && moveTo(e)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
      onKeyDown={onKeyDown}
    >
      <div className="absolute inset-0">{before}</div>
      <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        {after}
      </div>
      <CornerPill side="left">{beforeLabel}</CornerPill>
      <CornerPill side="right">{afterLabel}</CornerPill>
      <div
        className="pointer-events-none absolute inset-y-0 z-10 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.15)]"
        style={{ left: `${pos}%` }}
        aria-hidden
      >
        <span className="absolute top-1/2 left-1/2 flex size-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-neutral-800 shadow-lg ring-1 ring-black/10">
          <ChevronsLeftRight className="size-4" />
        </span>
      </div>
    </div>
  )
}
