import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import { toast } from "sonner"
import { api, eventsUrl, type Job } from "@/api"

type Live = { jobs: Record<string, Job>; lines: Record<string, string[]>; connected: boolean }

const RECONNECT_MS = 2000
const KEPT_LINES = 5000
const LiveContext = createContext<Live>({ jobs: {}, lines: {}, connected: false })

export function LiveProvider({ children }: { children: ReactNode }) {
  const [live, setLive] = useState<Live>({ jobs: {}, lines: {}, connected: false })
  useEffect(() => {
    let sock: WebSocket | null = null
    let stop = false
    const open = () => {
      sock = new WebSocket(eventsUrl())
      sock.onopen = () => setLive((l) => ({ ...l, connected: true }))
      sock.onclose = () => {
        setLive((l) => ({ ...l, connected: false }))
        if (!stop) setTimeout(open, RECONNECT_MS)
      }
      sock.onmessage = (m) => {
        const ev = JSON.parse(m.data)
        setLive((l) =>
          ev.type === "job"
            ? { ...l, jobs: { ...l.jobs, [ev.job.id]: ev.job } }
            : { ...l, lines: { ...l.lines, [ev.job]: [...(l.lines[ev.job] ?? []), ev.line].slice(-KEPT_LINES) } },
        )
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

export function useJobLines(id: string): string[] {
  const fresh = useLive().lines[id] ?? []
  const seen = useRef(fresh.length)
  seen.current = fresh.length
  const [backlog, setBacklog] = useState<{ lines: string[]; offset: number }>({ lines: [], offset: 0 })
  useEffect(() => {
    api<{ lines: string[] }>(`/jobs/${id}`)
      .then((j) => setBacklog({ lines: j.lines, offset: seen.current }))
      .catch(() => setBacklog({ lines: [], offset: 0 }))
  }, [id])
  return [...backlog.lines, ...fresh.slice(backlog.offset)]
}

const ENDED = new Set(["done", "failed", "cancelled"])

export function useJobToasts() {
  const { jobs } = useLive()
  const last = useRef<Record<string, Job["status"]>>({})
  useEffect(() => {
    for (const j of Object.values(jobs)) {
      const before = last.current[j.id]
      if (before && before !== j.status && ENDED.has(j.status)) {
        const title = `byosm ${j.args.slice(0, 2).join(" ")}`
        if (j.status === "done") toast.success(title, { description: "finished" })
        else toast.error(title, { description: j.status })
      }
      last.current[j.id] = j.status
    }
  }, [jobs])
}
