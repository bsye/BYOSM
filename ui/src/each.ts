import { useEffect, useRef, useState } from "react"
import { api } from "@/api"

const AT_ONCE = 4

export function useEach<T>(keys: string[], prefix: string, refresh: number): Record<string, T | null> {
  const [found, setFound] = useState<Record<string, T | null>>({})
  const fetched = useRef({ refresh, prefix, keys: new Set<string>() })
  const joined = keys.join("\n")
  useEffect(() => {
    if (fetched.current.refresh !== refresh || fetched.current.prefix !== prefix) fetched.current = { refresh, prefix, keys: new Set() }
    const have = fetched.current.keys
    const stop = new AbortController()
    const queue = joined.split("\n").filter((k) => k && !have.has(k))
    const work = async () => {
      for (let key = queue.shift(); key !== undefined && !stop.signal.aborted; key = queue.shift()) {
        const name = key
        const value = await api<T>(`${prefix}/${encodeURIComponent(name)}`, { signal: stop.signal }).catch(() => null)
        if (stop.signal.aborted) return
        have.add(name)
        setFound((cur) => ({ ...cur, [name]: value }))
      }
    }
    for (let i = 0; i < AT_ONCE; i++) void work()
    return () => stop.abort()
  }, [joined, prefix, refresh])
  return found
}
