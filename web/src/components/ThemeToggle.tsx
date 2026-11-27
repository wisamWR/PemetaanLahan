import { Monitor, Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useTheme, type ThemeChoice } from '@/lib/theme'

const NEXT: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' }
const LABEL: Record<ThemeChoice, string> = { system: 'Tema: ikuti sistem', light: 'Tema: terang', dark: 'Tema: gelap' }

export function ThemeToggle() {
  const { choice, setChoice } = useTheme()
  const Icon = choice === 'system' ? Monitor : choice === 'light' ? Sun : Moon
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon" aria-label={LABEL[choice]} onClick={() => setChoice(NEXT[choice])}>
          <Icon />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="right">{LABEL[choice]} (klik untuk ganti)</TooltipContent>
    </Tooltip>
  )
}
