import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import { api, eventsUrl, type Job } from "@/api"
import { announce } from "@/job-toast"

export type Metric = { kind: string; [key: string]: unknown }

type Live = {
  jobs: Record<string, Job>
  lines: Record<string, string[]>
  metrics: Record<string, Metric[]>
  received: Record<string, number>
  measured: number
  finished: Record<string, number>
  connected: boolean
  unauthorized: boolean
}

type LiveEvent = { type: string; job: Job & string; data?: Metric; line?: string }

const RECONNECT_MS = 2000
const FLUSH_MS = 100
const SEED_LINES = 50
export const UNFINISHED = new Set(["running", "queued"])
const ENDED = new Set(["done", "failed", "cancelled", "interrupted"])
const UNAUTHORIZED = 4401
const KEPT_LINES = 5000
const EMPTY: Live = { jobs: {}, lines: {}, metrics: {}, received: {}, measured: 0, finished: {}, connected: false, unauthorized: false }
const LiveContext = createContext<Live>(EMPTY)

function applied(l: Live, ev: LiveEvent, received: Record<string, number>): Live {
  if (ev.type === "job") {
    const before = l.jobs[ev.job.id]?.status
    const jobs = { ...l.jobs, [ev.job.id]: ev.job }
    if (before === undefined || !UNFINISHED.has(before) || !ENDED.has(ev.job.status)) return { ...l, jobs }
    const command = ev.job.args[0] ?? ""
    return { ...l, jobs, finished: { ...l.finished, [command]: (l.finished[command] ?? 0) + 1 } }
  }
  if (ev.type === "metric") return { ...l, metrics: { ...l.metrics, [ev.job]: [...(l.metrics[ev.job] ?? []), ev.data as Metric] } }
  if (ev.type === "measure") return { ...l, measured: l.measured + 1 }
  return {
    ...l,
    lines: { ...l.lines, [ev.job]: [...(l.lines[ev.job] ?? []), ev.line as string].slice(-KEPT_LINES) },
    received: { ...l.received, [ev.job]: received[ev.job] },
  }
}

export function LiveProvider({ children }: { children: ReactNode }) {
  const [live, setLive] = useState<Live>(EMPTY)
  useEffect(() => {
    let sock: WebSocket | null = null
    const seeded = new Set<string>()
    const received: Record<string, number> = {}
    let pending: LiveEvent[] = []
    let flush: ReturnType<typeof setTimeout> | null = null
    let stop = false
    const apply = () => {
      const batch = pending
      pending = []
      flush = null
      setLive((l) => batch.reduce((acc, ev) => applied(acc, ev, received), l))
    }
    const seed = (id: string) => {
      seeded.add(id)
      api<{ events?: Metric[]; lines: string[] }>(`/jobs/${id}`)
        .then((d) =>
          setLive((l) => ({
            ...l,
            metrics: { ...l.metrics, [id]: [...(d.events ?? []), ...(l.metrics[id] ?? [])] },
            lines: { ...l.lines, [id]: [...d.lines.slice(-SEED_LINES), ...(l.lines[id] ?? [])] },
          })),
        )
        .catch(() => seeded.delete(id))
    }
    const open = () => {
      sock = new WebSocket(eventsUrl())
      sock.onopen = () => setLive((l) => ({ ...l, connected: true, unauthorized: false }))
      sock.onclose = (e) => {
        setLive((l) => ({ ...l, connected: false, unauthorized: e.code === UNAUTHORIZED }))
        if (!stop) setTimeout(open, RECONNECT_MS)
      }
      sock.onmessage = (m) => {
        const ev: LiveEvent = JSON.parse(m.data)
        if (ev.type === "job" && UNFINISHED.has(ev.job.status) && !seeded.has(ev.job.id)) seed(ev.job.id)
        if (ev.type === "log") received[ev.job] = (received[ev.job] ?? 0) + 1
        pending.push(ev)
        flush ??= setTimeout(apply, FLUSH_MS)
      }
    }
    open()
    return () => {
      stop = true
      if (flush) clearTimeout(flush)
      sock?.close()
    }
  }, [])
  return <LiveContext.Provider value={live}>{children}</LiveContext.Provider>
}

export const useLive = () => useContext(LiveContext)

export function useFinished(command?: string): number {
  const { finished } = useLive()
  return command === undefined ? Object.values(finished).reduce((a, b) => a + b, 0) : (finished[command] ?? 0)
}

export function useJob(id: string): { lines: string[]; metrics: Metric[] } {
  const live = useLive()
  const fresh = { lines: live.lines[id] ?? [], metrics: live.metrics[id] ?? [] }
  const seen = useRef({ lines: 0, metrics: 0 })
  seen.current = { lines: live.received[id] ?? 0, metrics: fresh.metrics.length }
  const [stored, setStored] = useState({ lines: [] as string[], metrics: [] as Metric[], at: { lines: 0, metrics: 0 } })
  useEffect(() => {
    api<{ lines: string[]; events?: Metric[] }>(`/jobs/${id}`)
      .then((j) => setStored({ lines: j.lines, metrics: j.events ?? [], at: seen.current }))
      .catch(() => setStored({ lines: [], metrics: [], at: { lines: 0, metrics: 0 } }))
  }, [id])
  return {
    lines: [...stored.lines, ...since(fresh.lines, (live.received[id] ?? 0) - stored.at.lines)],
    metrics: unique([...stored.metrics, ...fresh.metrics.slice(stored.at.metrics)]),
  }
}

function since(lines: string[], count: number): string[] {
  return count > 0 ? lines.slice(Math.max(0, lines.length - count)) : []
}

function unique(metrics: Metric[]): Metric[] {
  const seen = new Set<string>()
  return metrics.filter((m) => {
    const key = JSON.stringify(m)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export function useJobToasts() {
  const { jobs } = useLive()
  const last = useRef<Record<string, Job["status"]>>({})
  useEffect(() => {
    for (const j of Object.values(jobs)) {
      const before = last.current[j.id]
      if (before && before !== j.status && ENDED.has(j.status)) {
        void announce(j)
      }
      last.current[j.id] = j.status
    }
  }, [jobs])
}
