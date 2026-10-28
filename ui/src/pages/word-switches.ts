import { useEffect, useState } from "react"
import { post } from "@/api"

export const pct = (v: number) => `${Math.round(v * 100)}%`

export function useWordSwitches<T extends { words: string[] }>(url: string, text: string, extra: (base: T) => object = () => ({})) {
  const [base, setBase] = useState<T | null>(null)
  const [now, setNow] = useState<T | null>(null)
  const [off, setOff] = useState<number[]>([])
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    setOff([])
    setBusy(true)
    post<T>(url, { input: text })
      .then((r) => {
        setBase(r)
        setNow(r)
      })
      .catch(() => setBase(null))
      .finally(() => setBusy(false))
  }, [url, text])
  const words = base?.words ?? []
  const kept = words.map((_, i) => i).filter((i) => !off.includes(i))
  const toggle = (i: number) => {
    if (!base) return
    const next = off.includes(i) ? off.filter((j) => j !== i) : [...off, i]
    setOff(next)
    setBusy(true)
    const input = words.filter((_, j) => !next.includes(j)).join(" ")
    post<T>(url, { input, ...extra(base) })
      .then(setNow)
      .finally(() => setBusy(false))
  }
  return { base, now, off, kept, removed: off.map((i) => words[i]), busy, toggle }
}
