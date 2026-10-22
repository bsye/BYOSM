import { CircleCheck, CircleSlash, CircleX, Clock, Loader2 } from "lucide-react"
import { useEffect, useState } from "react"
import { api, type Job, type Summary } from "@/api"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { took } from "@/format"
import { currentStep, describe, llmUsed, timeLeft } from "@/jobs"
import { openJob } from "@/job-sheet"
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
  const queued = rows.filter((r) => r.job.status === "queued").length
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm">
          <Loader2 className="animate-spin" />
          {rows.length - queued} running
          {queued > 0 && <span className="text-muted-foreground">+{queued} queued</span>}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[26rem]">
        {rows.map(({ job, now }) => {
          const d = describe(job)
          const waiting = job.status === "queued"
          return (
            <DropdownMenuItem key={job.id} onSelect={() => openJob(job.id)} className="items-start gap-3 py-2" title={d.subject ?? undefined}>
              {waiting ? <Clock className="text-muted-foreground mt-0.5" /> : <Loader2 className="mt-0.5 animate-spin" />}
              <div className="min-w-0 flex-1">
                <p className="truncate">
                  <span className="font-medium">{d.title}</span> <span className="text-foreground/70">{d.target ?? d.subject}</span>
                </p>
                <p className="text-muted-foreground truncate text-xs">{waiting ? "Queued: starts when memory frees up" : now}</p>
              </div>
              <span className="text-muted-foreground shrink-0 pt-0.5 text-xs tabular-nums">{waiting ? "" : took(job.started, null)}</span>
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

const STATUS_LABEL: Record<Job["status"], string> = { queued: "Queued", running: "Running", done: "Completed", failed: "Failed", cancelled: "Cancelled" }
const FILTERS = { all: () => true, running: (j: Job) => j.status === "running" || j.status === "queued", failed: (j: Job) => j.status === "failed", done: (j: Job) => j.status === "done" }
type Filter = keyof typeof FILTERS
const FILTER_LABELS: Record<Filter, string> = { all: "All", running: "Running", failed: "Failed", done: "Completed" }

function StatusBadge({ status }: { status: Job["status"] }) {
  return (
    <Badge variant={status === "failed" ? "destructive" : status === "done" ? "secondary" : "outline"}>
      <StatusIcon status={status} />
      {STATUS_LABEL[status]}
    </Badge>
  )
}

function useTitles(): Record<string, string> {
  const [titles, setTitles] = useState<Record<string, string>>({})
  useEffect(() => {
    api<Summary[]>("/specialists").then((s) => setTitles(Object.fromEntries(s.map((x) => [x.name, x.title || x.name]))), () => undefined)
  }, [])
  return titles
}

function JobList({ rows, titles }: { rows: ReturnType<typeof useRows>; titles: Record<string, string> }) {
  const { shown, footer } = useShowMore(rows)
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-32">Status</TableHead>
            <TableHead>Job</TableHead>
            <TableHead>Now</TableHead>
            <TableHead>Started</TableHead>
            <TableHead className="text-right">Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {shown.map(({ job, now, llm }) => {
            const d = describe(job)
            return (
              <TableRow key={job.id} className="cursor-pointer" onClick={() => go("jobs", job.id)}>
                <TableCell>
                  <StatusBadge status={job.status} />
                </TableCell>
                <TableCell className="max-w-md">
                  <div className="truncate font-medium" title={d.subject ?? undefined}>
                    {d.title} {d.target == null && d.subject && <span className="text-muted-foreground font-normal">{d.subject}</span>}
                  </div>
                  <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
                    {d.target && (
                      <a className="hover:underline" href={href("specialists", d.target)} onClick={(e) => e.stopPropagation()}>
                        {titles[d.target] ?? d.target}
                      </a>
                    )}
                    <span className="font-mono">{job.id.slice(0, 6)}</span>
                  </div>
                </TableCell>
                <TableCell className="text-muted-foreground max-w-xs text-xs">
                  <div className="truncate">{now ?? "-"}</div>
                  {llm && <div className="text-muted-foreground/70 truncate">{llm}</div>}
                </TableCell>
                <TableCell className="text-foreground/70 text-xs tabular-nums">{started(job.started)}</TableCell>
                <TableCell className="text-foreground/70 text-right tabular-nums">{took(job.started, job.ended)}</TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
      {footer}
    </>
  )
}

const started = (t: number | null) =>
  t == null ? "-" : new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })

export function Jobs({ id }: { id?: string }) {
  const rows = useRows()
  const titles = useTitles()
  const [filter, setFilter] = useState<Filter>("all")
  if (id) return <JobDetail id={id} />
  const count = (f: Filter) => rows.filter((r) => FILTERS[f](r.job)).length
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Jobs</h1>
      <Timeline jobs={rows.map((r) => r.job)} titles={titles} />
      <Card>
        <CardHeader>
          <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
            <TabsList>
              {(Object.keys(FILTERS) as Filter[]).map((f) => (
                <TabsTrigger key={f} value={f}>
                  {FILTER_LABELS[f]} <span className="text-muted-foreground tabular-nums">{count(f)}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          <JobList key={filter} rows={rows.filter((r) => FILTERS[filter](r.job))} titles={titles} />
        </CardContent>
      </Card>
    </div>
  )
}
