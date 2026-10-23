import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import { api, eventsUrl, type Job } from "@/api"
import { announce } from "@/job-toast"

export type Metric = { kind: string; [key: string]: unknown }

type Live = {
  jobs: Record<string, Job>
  lines: Record<string, string[]>
  metrics: Record<string, Metric[]>
  received: Record<string, number>
  connected: boolean
  unauthorized: boolean
}

const RECONNECT_MS = 2000
const SEED_LINES = 50
const UNFINISHED = new Set(["running", "queued"])
const UNAUTHORIZED = 4401
const KEPT_LINES = 5000
const EMPTY: Live = { jobs: {}, lines: {}, metrics: {}, received: {}, connected: false, unauthorized: false }
const LiveContext = createContext<Live>(EMPTY)

export function LiveProvider({ children }: { children: ReactNode }) {
  const [live, setLive] = useState<Live>(EMPTY)
  useEffect(() => {
    let sock: WebSocket | null = null
    const seeded = new Set<string>()
    const received: Record<string, number> = {}
    let stop = false
    const open = () => {
      sock = new WebSocket(eventsUrl())
      sock.onopen = () => setLive((l) => ({ ...l, connected: true, unauthorized: false }))
      sock.onclose = (e) => {
        setLive((l) => ({ ...l, connected: false, unauthorized: e.code === UNAUTHORIZED }))
        if (!stop) setTimeout(open, RECONNECT_MS)
      }
      sock.onmessage = (m) => {
        const ev = JSON.parse(m.data)
        if (ev.type === "job" && UNFINISHED.has(ev.job.status) && !seeded.has(ev.job.id)) {
          seeded.add(ev.job.id)
          const from = received[ev.job.id] ?? 0
          api<{ events?: Metric[]; lines: string[] }>(`/jobs/${ev.job.id}`)
            .then((d) =>
              setLive((l) => ({
                ...l,
                metrics: { ...l.metrics, [ev.job.id]: [...(d.events ?? []), ...(l.metrics[ev.job.id] ?? [])] },
                lines: {
                  ...l.lines,
                  [ev.job.id]: [...d.lines.slice(-SEED_LINES), ...since(l.lines[ev.job.id] ?? [], (received[ev.job.id] ?? 0) - from)],
                },
              })),
            )
            .catch(() => seeded.delete(ev.job.id))
        }
        const count = ev.type === "log" ? (received[ev.job] = (received[ev.job] ?? 0) + 1) : 0
        setLive((l) => {
          if (ev.type === "job") return { ...l, jobs: { ...l.jobs, [ev.job.id]: ev.job } }
          if (ev.type === "metric") return { ...l, metrics: { ...l.metrics, [ev.job]: [...(l.metrics[ev.job] ?? []), ev.data] } }
          return {
            ...l,
            lines: { ...l.lines, [ev.job]: [...(l.lines[ev.job] ?? []), ev.line].slice(-KEPT_LINES) },
            received: { ...l.received, [ev.job]: count },
          }
        })
      }
    }
    open()
    return () => {
      stop = true
      sock?.close()
    }
  }, [])
  return <LiveContext.Provider value={live}>{children}</LiveContext.Provider>
}

export const useLive = () => useContext(LiveContext)

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

const ENDED = new Set(["done", "failed", "cancelled", "interrupted"])

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
