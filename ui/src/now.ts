import { useEffect, useState } from "react"

const TICK_MS = 1000

export function useNow(active: boolean): number {
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now() / 1000), TICK_MS)
    return () => clearInterval(t)
  }, [active])
  return now
}
