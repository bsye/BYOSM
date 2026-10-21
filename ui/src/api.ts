const TOKEN_KEY = "byosm-token"

export function readToken(): string {
  const fromHash = new URLSearchParams(window.location.hash.slice(1)).get("token")
  if (fromHash) {
    sessionStorage.setItem(TOKEN_KEY, fromHash)
    history.replaceState(null, "", window.location.pathname)
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? ""
}

export function saveToken(value: string) {
  sessionStorage.setItem(TOKEN_KEY, value.includes("token=") ? value.slice(value.indexOf("token=") + 6) : value.trim())
  window.location.reload()
}

async function call(path: string, init: RequestInit = {}): Promise<Response> {
  const json: Record<string, string> = init.body instanceof FormData ? {} : { "Content-Type": "application/json" }
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { ...json, Authorization: `Bearer ${readToken()}`, ...init.headers },
  })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText)
  return res
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  return (await call(path, init)).json().catch(() => ({})) as Promise<T>
}

export async function download(path: string, filename: string) {
  const url = URL.createObjectURL(await (await call(path)).blob())
  const link = Object.assign(document.createElement("a"), { href: url, download: filename })
  link.click()
  URL.revokeObjectURL(url)
}

export function upload<T>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append("file", file)
  return api<T>(path, { method: "POST", body: form })
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
  benchmark: string
  promoted: string | null
  score: number | null
  rival: { name: string; score: number | null } | null
  experiments: string[]
  exported: boolean
}

export type Run = {
  name: string
  parent: string | null
  params: number | null
  n_train: number | null
  train_seconds: number | null
  best_epoch: number | null
  history: { epoch: number; loss: number; dev: number }[]
}

export type SplitSummary = { n: number; labels: Record<string, number> }

export type Detail = Summary & {
  spec: Record<string, unknown>
  board: Record<string, Record<string, Result>>
  baselines: string[]
  runs: Run[]
  data: Record<string, SplitSummary> | null
  proposals: { from: string; proposals: Proposal[] } | null
}

export type Proposal = {
  id: string
  why: string
  ops: { op: string; path: string; value: unknown }[]
  cost?: { minutes: number; new_items: number }
}

export type Need = { train_min: number; train_recommended: number; test_needed: number }

export type DataRequest = {
  request: { reason: string; counts: Record<string, Need> } | null
  templates: string[]
  uploads: string[]
}

export type Suspect = { row: number; given: string; probe: string; llm?: string; why?: string }

export type CheckReport = {
  file: string
  verdict: "ready" | "needs more" | "fix"
  coverage: Record<string, Partial<Need> & { rows: number; test: number }>
  shortfalls: string[]
  issues: [number, string, string][]
  noise: { agreement: number; confusions: { given: string; probe: string; rows: number }[]; suspects: Suspect[] } | null
}

export type ReviewRow = {
  input: string
  panel: string
  sure: number
  votes: Record<string, string | null>
  reviewed: string | null
}

export type PanelView = {
  teachers: string[]
  labels: string[]
  summary: {
    teachers: Record<string, number>
    panel_on_labelled: number
    n_labelled: number
    n_pool: number
    all_agree: number
  } | null
  n_reviewed: number
  review: ReviewRow[]
}

export const runJob = (args: string[]) => post<Job>("/jobs", { args })

export function eventsUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws"
  return `${scheme}://${window.location.host}/api/events?token=${encodeURIComponent(readToken())}`
}
