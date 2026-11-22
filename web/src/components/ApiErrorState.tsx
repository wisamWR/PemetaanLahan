import { TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

/** Galat seluruh halaman (API tidak terjangkau): pesan jelas, tombol coba lagi, rincian teknis (PayPal, Typeform, Tines). */
export function ApiErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <Card className="border-dashed" role="alert" data-testid="api-error">
      <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10 text-destructive">
          <TriangleAlert className="size-6" />
        </div>
        <h2 className="text-lg font-semibold">API tidak bisa dihubungi</h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Halaman ini membaca hasil dari server API. Pastikan server berjalan (lihat web/README.md), lalu coba lagi.
        </p>
        <p className="max-w-md rounded-lg bg-muted px-3 py-2 font-mono text-xs break-words text-muted-foreground">{message}</p>
        <Button className="mt-1" onClick={onRetry}>
          Coba lagi
        </Button>
      </CardContent>
    </Card>
  )
}
