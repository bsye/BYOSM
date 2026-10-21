import { CircleCheck, CircleX, Loader2, Square } from "lucide-react"
import { useEffect, useRef } from "react"
import { post, type Job } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useJob, useLive } from "@/events"
import { took } from "@/format"
import { llmUsed, steps, type Step } from "@/jobs"
import { JobMetrics } from "./JobMetrics"
import { JobTitle, StatusIcon } from "./Jobs"

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

const STEP_ICON = { running: Loader2, done: CircleCheck, failed: CircleX }

function Steps({ items }: { items: Step[] }) {
  if (!items.length) return <p className="text-muted-foreground text-sm">No steps reported yet.</p>
  return (
    <ol className="divide-y">
      {items.map((s, i) => {
        const Icon = STEP_ICON[s.state]
        return (
          <li key={i} className="flex items-center gap-3 py-2 text-sm">
            <Icon className={`size-4 shrink-0 ${s.state === "running" ? "animate-spin" : s.state === "failed" ? "text-destructive" : "text-foreground/70"}`} />
            <span className="font-medium">{s.label}</span>
            <span className="text-muted-foreground ml-auto text-xs tabular-nums">{s.detail}</span>
          </li>
        )
      })}
    </ol>
  )
}

function Log({ lines }: { lines: string[] }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" })
  }, [lines.length])
  return (
    <div className="bg-muted h-[28rem] overflow-auto rounded-md p-3 font-mono text-xs leading-5 whitespace-pre">
      {lines.map((l, i) => (
        <div key={i}>{l}</div>
      ))}
      <div ref={end} />
    </div>
  )
}

export function JobDetail({ id }: { id: string }) {
  const job = useLive().jobs[id]
  const { lines, metrics } = useJob(id)
  if (!job) return <p className="text-muted-foreground text-sm">No job {id} on this server.</p>
  const active = job.status === "running" || job.status === "queued"
  const llm = llmUsed(metrics)
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <StatusIcon status={job.status} />
            <JobTitle job={job} />
          </h1>
          <p className="text-muted-foreground/70 text-xs tabular-nums">
            {job.status}, {took(job.started, job.ended)}
            {llm && `, LLM ${llm}`}
          </p>
        </div>
        {active && (
          <Button variant="outline" onClick={() => post(`/jobs/${id}/cancel`, {})}>
            <Square />
            Cancel
          </Button>
        )}
      </div>
      <JobMetrics metrics={metrics} />
      <Tabs defaultValue="steps">
        <TabsList>
          <TabsTrigger value="steps">Steps</TabsTrigger>
          <TabsTrigger value="log">Log</TabsTrigger>
        </TabsList>
        <TabsContent value="steps">
          <Card>
            <CardContent>
              <Steps items={steps(metrics, active)} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="log">
          <Log lines={lines} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
