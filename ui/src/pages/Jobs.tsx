import { CircleCheck, CircleSlash, CircleX, Clock, Loader2 } from "lucide-react"
import type { Job } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { took, when } from "@/format"
import { currentStep, describe, llmUsed, timeLeft } from "@/jobs"
import { go, href } from "@/route"
import { JobDetail } from "./JobLog"
import { Timeline } from "./Timeline"

const ICONS = { queued: Clock, running: Loader2, done: CircleCheck, failed: CircleX, cancelled: CircleSlash }
const TONE = {
  queued: "text-muted-foreground",
  running: "text-foreground animate-spin",
  done: "text-foreground/70",
  failed: "text-destructive",
  cancelled: "text-muted-foreground",
}

export function StatusIcon({ status }: { status: Job["status"] }) {
  const Icon = ICONS[status]
  return <Icon className={`size-4 shrink-0 ${TONE[status]}`} aria-label={status} />
}

export function JobTitle({ job }: { job: Job }) {
  const d = describe(job)
  return (
    <span className="flex min-w-0 items-baseline gap-1.5">
      <span className="font-medium">{d.title}</span>
      {d.target && (
        <a className="hover:underline" href={href("specialists", d.target)} onClick={(e) => e.stopPropagation()}>
          {d.target}
        </a>
      )}
      {d.subject && <span className="text-muted-foreground truncate">{d.subject}</span>}
    </span>
  )
}

function useRows() {
  const live = useLive()
  return Object.values(live.jobs)
    .sort((a, b) => b.created - a.created)
    .map((j) => {
      const metrics = live.metrics[j.id] ?? []
      const active = j.status === "running" || j.status === "queued"
      const left = active ? timeLeft(metrics) : null
      const step = active ? currentStep(metrics, live.lines[j.id] ?? []) : null
      return { job: j, now: step && left ? `${step}, ${left}` : step, llm: llmUsed(metrics) }
    })
}

export function Running() {
  const rows = useRows().filter((r) => r.now != null)
  if (!rows.length) return null
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm">
          <Loader2 className="animate-spin" />
          {rows.length} running
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-96">
        {rows.map(({ job, now }) => (
          <DropdownMenuItem key={job.id} onSelect={() => go("jobs", job.id)} className="flex-col items-start gap-0.5">
            <JobTitle job={job} />
            <span className="text-muted-foreground w-full truncate text-xs">
              {now}, {took(job.started, job.ended)}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function Jobs({ id }: { id?: string }) {
  const rows = useRows()
  if (id) return <JobDetail id={id} />
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Jobs</h1>
      <Card>
        <CardHeader>
          <CardTitle>All jobs</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8" />
                <TableHead>Job</TableHead>
                <TableHead>Now</TableHead>
                <TableHead>LLM</TableHead>
                <TableHead>Started</TableHead>
                <TableHead className="text-right">Took</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(({ job, now, llm }) => (
                <TableRow key={job.id} className="cursor-pointer" onClick={() => go("jobs", job.id)}>
                  <TableCell>
                    <StatusIcon status={job.status} />
                  </TableCell>
                  <TableCell className="max-w-md">
                    <JobTitle job={job} />
                  </TableCell>
                  <TableCell className="text-muted-foreground max-w-xs truncate text-xs">{now ?? job.status}</TableCell>
                  <TableCell className="text-foreground/70 text-xs">{llm ?? "-"}</TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{when(job.started)}</TableCell>
                  <TableCell className="text-foreground/70 text-right tabular-nums">{took(job.started, job.ended)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Timeline jobs={rows.map((r) => r.job)} />
    </div>
  )
}
