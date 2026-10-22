import { Hourglass, Laptop, RotateCw } from "lucide-react"
import { useEffect, useState } from "react"
import { api, post, put } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useLive, type Metric } from "@/events"

const LIMIT = "limit"
const ACTIVE = new Set(["running", "queued"])

const clock = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

function waitingUntil(metrics: Metric[]): number | null {
  const last = metrics.findLast((m) => m.kind === LIMIT)
  return last && last.until != null && last.resumed == null ? Number(last.until) : null
}

export function useLimitWaits(): { jobs: number; until: number | null; reason: string } {
  const live = useLive()
  const waits = Object.values(live.jobs)
    .filter((j) => ACTIVE.has(j.status))
    .map((j) => ({ until: waitingUntil(live.metrics[j.id] ?? []), metrics: live.metrics[j.id] ?? [] }))
    .filter((w) => w.until != null)
  const until = waits.length ? Math.min(...waits.map((w) => w.until as number)) : null
  const reason = String(waits[0]?.metrics.findLast((m) => m.kind === LIMIT)?.error ?? "")
  return { jobs: waits.length, until, reason }
}

export function LimitBanner() {
  const { jobs, until, reason } = useLimitWaits()
  const [busy, setBusy] = useState(false)
  if (!jobs || until == null) return null
  const act = (fn: () => Promise<unknown>) => {
    setBusy(true)
    void fn().finally(() => setBusy(false))
  }
  return (
    <Alert>
      <Hourglass />
      <AlertTitle>
        LLM limit reached: {jobs} {jobs === 1 ? "job waits" : "jobs wait"} until {clock(until)}, then retry on their own
      </AlertTitle>
      <AlertDescription className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 truncate" title={reason}>
          {reason}
        </span>
        <Button size="sm" variant="outline" disabled={busy} onClick={() => act(() => post("/llm/on-limit/retry", {}))}>
          <RotateCw />
          Retry now
        </Button>
        <Button size="sm" disabled={busy} onClick={() => act(() => put("/llm/on-limit", { mode: "local" }))}>
          <Laptop />
          Switch to local
        </Button>
      </AlertDescription>
    </Alert>
  )
}

const MODE_LABELS: Record<string, string> = { wait: "Wait and retry", local: "Switch to local", fail: "Stop the job" }

export function OnLimitSelect() {
  const [mode, setMode] = useState<string | null>(null)
  useEffect(() => {
    api<{ mode: string }>("/llm/on-limit").then((s) => setMode(s.mode), () => setMode(null))
  }, [])
  const pick = (m: string) => {
    setMode(m)
    void put("/llm/on-limit", { mode: m })
  }
  return (
    <div className="space-y-2">
      <Label htmlFor="on-limit">When the LLM hits its limit</Label>
      <Select value={mode ?? undefined} onValueChange={pick}>
        <SelectTrigger id="on-limit" size="sm" className="w-44">
          <SelectValue placeholder="Loading..." />
        </SelectTrigger>
        <SelectContent>
          {Object.entries(MODE_LABELS).map(([k, v]) => (
            <SelectItem key={k} value={k}>
              {v}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
