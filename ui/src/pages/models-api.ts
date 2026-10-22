import { useCallback, useEffect, useState } from "react"
import { api } from "@/api"
import { useLive } from "@/events"

export type Usage = { job: string; role: string; model: string; calls: number; failed: number; tokens_in: number; tokens_out: number; seconds: number; last: number }
export type Loaded = { server: string; model: string; bytes: number | null; params: string | null; quantization: string | null; until: string | null }
export type Role = { role: string; own: boolean; provider: string | null; model: string | null; base_url: string | null; effort: string | null; effort_levels: string[]; max_parallel: number | null }
export type Roles = { roles: Role[]; budget: { tokens_per_job: number | null } }

export function useModels() {
  const [usage, setUsage] = useState<Usage[]>([])
  const [loaded, setLoaded] = useState<Loaded[] | null>(null)
  const [roles, setRoles] = useState<Roles | null>(null)
  const live = useLive()
  const calls = Object.values(live.metrics).reduce((n, ms) => n + ms.filter((m) => m.kind === "llm" && m.state !== "waiting").length, 0)
  const refresh = useCallback(() => {
    api<Usage[]>("/llm/usage").then(setUsage).catch(() => setUsage([]))
    api<Loaded[]>("/llm/local").then(setLoaded).catch(() => setLoaded([]))
    api<Roles>("/llm/roles").then(setRoles).catch(() => setRoles(null))
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh, calls])
  return { usage, loaded, setLoaded, roles, setRoles, refresh }
}

export function byTaskAndModel(usage: Usage[]): (Usage & { jobs: number })[] {
  const out = new Map<string, Usage & { jobs: number }>()
  for (const u of usage) {
    const key = `${u.role}|${u.model}`
    const cur = out.get(key)
    out.set(
      key,
      cur
        ? { ...cur, calls: cur.calls + u.calls, failed: cur.failed + u.failed, tokens_in: cur.tokens_in + u.tokens_in, tokens_out: cur.tokens_out + u.tokens_out, seconds: cur.seconds + u.seconds, last: Math.max(cur.last, u.last), jobs: cur.jobs + 1 }
        : { ...u, jobs: 1 },
    )
  }
  return [...out.values()].sort((a, b) => b.tokens_in + b.tokens_out - (a.tokens_in + a.tokens_out) || b.calls - a.calls)
}
