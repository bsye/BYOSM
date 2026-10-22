import { CircleX, Laptop, Loader2, RotateCw, Square } from "lucide-react"
import { useEffect, useRef } from "react"
import { api, post, runJob } from "@/api"
import { openJob } from "@/job-sheet"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useJob, useLive } from "@/events"
import { seconds, took } from "@/format"
import { decisions, stages } from "@/job-story"
import { currentStep, describe, EVENT_PREFIX, failure, llmUsed, steps, timeLeft } from "@/jobs"
import { useNow } from "@/now"
import { href } from "@/route"
import { DecisionTree } from "./DecisionTree"
import { JobMetrics } from "./JobMetrics"
import { JobStages } from "./JobStages"
import { StatusIcon } from "./Jobs"

const LOG_TAIL = 2000

function Now({ text, since, left, now }: { text: string; since: number | null; left: string | null; now: number }) {
  return (
    <div className="bg-muted/40 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border px-4 py-3">
      <Loader2 className="size-4 shrink-0 animate-spin" />
      <span className="min-w-0 flex-1 truncate font-medium">{text}</span>
      <span className="text-muted-foreground text-xs tabular-nums">
        {since != null && `last update ${seconds(Math.max(0, now - since))} ago`}
        {left && ` · ${left}`}
      </span>
    </div>
  )
}

function RetryButtons({ args }: { args: string[] }) {
  const retry = async (local: boolean) => {
    const llm = local ? (await api<{ local: Record<string, unknown> }>("/llm/on-limit")).local : null
    openJob((await runJob(args, llm)).id)
  }
  return (
    <div className="flex flex-wrap gap-2">
      <Button size="sm" variant="outline" onClick={() => retry(false)}>
        <RotateCw />
        Retry
      </Button>
      <Button size="sm" variant="outline" onClick={() => retry(true)}>
        <Laptop />
        Retry with local LLM
      </Button>
    </div>
  )
}

function Log({ lines }: { lines: string[] }) {
  const end = useRef<HTMLDivElement>(null)
  const shown = lines.filter((l) => !l.startsWith(EVENT_PREFIX)).slice(-LOG_TAIL)
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" })
  }, [shown.length])
  return (
    <Card>
      <CardHeader>
        <CardTitle>Log</CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">{shown.length} lines</CardAction>
      </CardHeader>
      <CardContent>
        <div className="bg-muted/50 h-96 overflow-auto rounded-md py-2 font-mono text-xs leading-5">
          {shown.map((l, i) => (
            <div key={i} className="hover:bg-muted flex gap-3 px-3">
              <span className="text-muted-foreground/50 w-8 shrink-0 text-right tabular-nums select-none">{i + 1}</span>
              <span className="whitespace-pre-wrap break-all">{l}</span>
            </div>
          ))}
          {!shown.length && <p className="text-muted-foreground px-3">Waiting for the first line.</p>}
          <div ref={end} />
        </div>
      </CardContent>
    </Card>
  )
}

export function JobDetail({ id }: { id: string }) {
  const job = useLive().jobs[id]
  const { lines, metrics } = useJob(id)
  const active = job?.status === "running" || job?.status === "queued"
  const now = useNow(active)
  if (!job) return <p className="text-muted-foreground text-sm">No job {id} on this server.</p>
  const llm = llmUsed(metrics)
  const story = stages(metrics, steps(metrics, active))
  const choices = decisions(metrics)
  const last = metrics.at(-1)?.at
  const d = describe(job)
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="flex items-start gap-2 text-2xl font-semibold">
            <span className="pt-2">
              <StatusIcon status={job.status} />
            </span>
            <span className="min-w-0">
              {d.title}{" "}
              {d.target && (
                <a className="text-muted-foreground hover:underline" href={href("specialists", d.target)}>
                  {d.target}
                </a>
              )}
              {d.subject && <span className="text-muted-foreground block truncate text-sm font-normal">{d.subject}</span>}
            </span>
          </h1>
          <p className="text-muted-foreground/70 text-xs tabular-nums">
            {job.status}, {active && job.started ? seconds(now - job.started) : took(job.started, job.ended)}
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
      {job.status === "failed" && (
        <Alert variant="destructive">
          <CircleX />
          <AlertTitle>Failed</AlertTitle>
          <AlertDescription className="space-y-3">
            <p className="font-mono text-xs break-all">{failure(lines) ?? "The job ended without output."}</p>
            <RetryButtons args={job.args[0] === "auto" && job.target ? ["auto", job.target] : job.args} />
          </AlertDescription>
        </Alert>
      )}
      {active && <Now text={currentStep(metrics, lines)} since={last == null ? job.started : Number(last)} left={timeLeft(metrics)} now={now} />}
      <div className={`grid gap-4 md:gap-6 ${choices.length ? "@5xl/main:grid-cols-2" : ""}`}>
        <JobStages items={story} started={job.started} active={active} now={now} />
        {choices.length > 0 && <DecisionTree items={choices} />}
      </div>
      <JobMetrics metrics={metrics} />
      <Log lines={lines} />
    </div>
  )
}
