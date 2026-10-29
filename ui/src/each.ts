import { useEffect, useState } from "react"
import { api } from "@/api"

const AT_ONCE = 4

export function useEach<T>(keys: string[], prefix: string, refresh: number): Record<string, T | null> {
  const [found, setFound] = useState<Record<string, T | null>>({})
  const joined = keys.join("\n")
  useEffect(() => {
    const stop = new AbortController()
    const queue = joined.split("\n").filter(Boolean)
    const work = async () => {
      for (let key = queue.shift(); key !== undefined && !stop.signal.aborted; key = queue.shift()) {
        const name = key
        const value = await api<T>(`${prefix}/${encodeURIComponent(name)}`, { signal: stop.signal }).catch(() => null)
        if (!stop.signal.aborted) setFound((cur) => ({ ...cur, [name]: value }))
      }
    }
    for (let i = 0; i < AT_ONCE; i++) void work()
    return () => stop.abort()
  }, [joined, prefix, refresh])
  return found
}
