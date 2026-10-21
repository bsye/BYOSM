import { useEffect, useRef } from "react"
import { post, type Job } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useJob, useLive } from "@/events"
import { took } from "@/format"
import { JobMetrics } from "./JobMetrics"

const STATUS_VARIANT: Record<Job["status"], "default" | "secondary" | "destructive" | "outline"> = {
  queued: "outline",
  running: "default",
  done: "secondary",
  failed: "destructive",
  cancelled: "outline",
}

export function StatusBadge({ status }: { status: Job["status"] }) {
  return <Badge variant={STATUS_VARIANT[status]}>{status}</Badge>
}

export function JobLog({ id }: { id: string }) {
  const job = useLive().jobs[id]
  const { lines, metrics } = useJob(id)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => end.current?.scrollIntoView({ block: "nearest" }), [lines.length])
  if (!job) return <p className="text-muted-foreground text-sm">Job {id} is not known to this server.</p>
  const active = job.status === "running" || job.status === "queued"
  return (
    <div className="space-y-4">
    <JobMetrics metrics={metrics} />
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-4">
        <CardTitle className="font-mono text-sm break-all">byosm {job.args.join(" ")}</CardTitle>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground text-xs">{took(job.started, job.ended)}</span>
          <StatusBadge status={job.status} />
          {active && (
            <Button size="sm" variant="outline" onClick={() => post(`/jobs/${id}/cancel`, {})}>
              Cancel
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="bg-muted h-[28rem] overflow-auto rounded-md p-3 font-mono text-xs leading-5 whitespace-pre">
          {lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
          <div ref={end} />
        </div>
      </CardContent>
    </Card>
    </div>
  )
}
