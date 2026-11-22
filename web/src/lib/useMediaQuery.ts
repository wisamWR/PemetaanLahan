import { useEffect, useState } from 'react'

/** True bila `query` (mis. `(min-width: 768px)`) cocok; mengikuti perubahan ukuran jendela. Aman tanpa matchMedia. */
export function useMediaQuery(query: string): boolean {
  const read = () => typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia(query).matches
  const [matches, setMatches] = useState(read)
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mql = window.matchMedia(query)
    const onChange = () => setMatches(mql.matches)
    onChange()
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [query])
  return matches
}
