const TOKEN_KEY = "byosm-token"

export function readToken(): string {
  const fromHash = new URLSearchParams(window.location.hash.slice(1)).get("token")
  if (fromHash) {
    sessionStorage.setItem(TOKEN_KEY, fromHash)
    history.replaceState(null, "", window.location.pathname)
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? ""
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${readToken()}`, ...init.headers },
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail ?? res.statusText)
  return body as T
}

export const post = <T>(path: string, body: unknown) => api<T>(path, { method: "POST", body: JSON.stringify(body) })
export const put = <T>(path: string, body: unknown) => api<T>(path, { method: "PUT", body: JSON.stringify(body) })
export const del = <T>(path: string) => api<T>(path, { method: "DELETE" })

export type Job = {
  id: string
  args: string[]
  status: "queued" | "running" | "done" | "failed" | "cancelled"
  created: number
  started: number | null
  ended: number | null
  code: number | null
}

export type Result = {
  objective: number | null
  calib_objective: number | null
  latency_ms: number | null
  params: number | null
  n_test: number | null
  ci: { lo: number; hi: number } | null
}

export type Summary = {
  name: string
  description: string
  modality: string
  task_type: string
  labels: number
  metric: string
  primary_suite: string
  promoted: string | null
  score: number | null
  rival: { name: string; score: number | null } | null
  experiments: string[]
  exported: boolean
}

export type Detail = Summary & {
  spec: Record<string, unknown>
  board: Record<string, Record<string, Result>>
  baselines: string[]
}

export const runJob = (args: string[]) => post<Job>("/jobs", { args })

export function eventsUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws"
  return `${scheme}://${window.location.host}/api/events?token=${encodeURIComponent(readToken())}`
}
