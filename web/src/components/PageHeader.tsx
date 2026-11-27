import type { ReactNode } from 'react'

/**
 * Kepala halaman: judul besar di area isi, keterangan redup di bawahnya, aksi utama di kanan
 * (pola Ferndesk "Overview", Sentry "Compare Queries", AirOps/Profound "Analytics").
 *
 * Di layar sempit (< md) aksi utama pindah ke bilah tetap di dasar layar dengan tombol selebar baris, supaya tetap
 * terjangkau ibu jari tanpa menggulir ke atas (pola eBay dan Google Health di iOS: tombol utama menempel di bawah).
 * `App` memberi ruang di dasar halaman (`pb-24`) agar bilah tidak menutupi isi.
 */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
      <div className="min-w-0 flex-1 basis-72">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && (
        <div
          data-testid="page-actions"
          className="fixed inset-x-0 bottom-0 z-30 flex items-center gap-2 border-t bg-background/95 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur *:flex-1 md:static md:z-auto md:flex-wrap md:border-0 md:bg-transparent md:p-0 md:backdrop-blur-none md:*:flex-none"
        >
          {actions}
        </div>
      )}
    </div>
  )
}
