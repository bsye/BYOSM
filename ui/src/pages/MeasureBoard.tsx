import { CircleCheck, CircleDashed, CircleX, Clock, Gauge, Loader2, TriangleAlert } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { toast } from "sonner"
import { api, post, type Job } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import { href } from "@/route"

type Status = "todo" | "pending" | "running" | "done" | "failed"

type Cell = {
  specialist: string
  measurement: string
  title: string
  why: string
  job: string | null
  status: Status
  note: string | null
}

type Board = {
  measurements: { id: string; title: string }[]
  cells: Cell[]
  progress: { done: number; failed: number; todo: number; total: number }
  error: string | null
}

export type Measure = { board: Board | null; failed: string | null; start: () => Promise<void>; starting: boolean }

const STATES = {
  todo: { icon: CircleDashed, label: "To do", tone: "text-muted-foreground" },
  pending: { icon: Clock, label: "Waiting", tone: "text-muted-foreground" },
  running: { icon: Loader2, label: "Running", tone: "animate-spin" },
  done: { icon: CircleCheck, label: "Done", tone: "text-foreground/70" },
  failed: { icon: CircleX, label: "Failed", tone: "text-destructive" },
} satisfies Record<Status, { icon: typeof Clock; label: string; tone: string }>

const UNFINISHED = new Set<Status>(["pending", "running"])

function live(cell: Cell, jobs: Record<string, Job>): Status {
  const job = cell.job ? jobs[cell.job] : undefined
  if (!UNFINISHED.has(cell.status) || !job) return cell.status
  return job.status === "queued" ? "pending" : "running"
}

export function useMeasure(): Measure {
  const { measured } = useLive()
  const [board, setBoard] = useState<Board | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const load = useCallback(
    () =>
      api<Board>("/measure").then(
        (b) => {
          setBoard(b)
          setFailed(null)
        },
        (e: Error) => setFailed(e.message),
      ),
    [],
  )
  useEffect(() => void load(), [load, measured])
  const start = async () => {
    setStarting(true)
    try {
      const b = await post<Board>("/measure", {})
      setBoard(b)
      if (!b.progress.todo && !b.progress.failed && !b.cells.some((c) => UNFINISHED.has(c.status))) toast("Every promoted model is measured")
    } catch (e) {
      toast.error("Could not start the measurements", { description: String(e) })
    } finally {
      setStarting(false)
    }
  }
  return { board, failed, start, starting }
}

export function MeasureButton({ measure }: { measure: Measure }) {
  return (
    <Button onClick={() => void measure.start()}>
      {measure.starting ? <Loader2 className="animate-spin" /> : <Gauge />}
      Measure everything
    </Button>
  )
}

function State({ cell, status }: { cell: Cell; status: Status }) {
  const { icon: Icon, label, tone } = STATES[status]
  const body = (
    <>
      <Icon className={`size-4 shrink-0 ${tone}`} />
      {label}
    </>
  )
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {cell.job ? (
          <Button variant="ghost" size="sm" className="-ml-2" onClick={() => openJob(cell.job ?? "")}>
            {body}
          </Button>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-sm">{body}</span>
        )}
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{cell.note ? `${cell.note}. ${cell.why}` : cell.why}</TooltipContent>
    </Tooltip>
  )
}

function Grid({ board, titles }: { board: Board; titles: Record<string, string> }) {
  const { jobs } = useLive()
  const shown = board.measurements.filter((m) => board.cells.some((c) => c.measurement === m.id))
  const rows = [...new Set(board.cells.map((c) => c.specialist))]
  const at = (who: string, what: string) => board.cells.find((c) => c.specialist === who && c.measurement === what)
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Specialist</TableHead>
          {shown.map((m) => (
            <TableHead key={m.id}>{m.title}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((who) => (
          <TableRow key={who}>
            <TableCell className="font-medium">
              <a className="hover:underline" href={href("specialists", who)}>
                {titles[who] ?? who}
              </a>
            </TableCell>
            {shown.map((m) => {
              const cell = at(who, m.id)
              return (
                <TableCell key={m.id}>{cell ? <State cell={cell} status={live(cell, jobs)} /> : <span className="text-muted-foreground/70">-</span>}</TableCell>
              )
            })}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function MeasureBoard({ measure, titles }: { measure: Measure; titles: Record<string, string> }) {
  const { board, failed } = measure
  if (failed)
    return (
      <Alert variant="destructive">
        <TriangleAlert />
        <AlertTitle>Could not load the measurements</AlertTitle>
        <AlertDescription>{failed}</AlertDescription>
      </Alert>
    )
  if (!board) return <Skeleton className="h-48 rounded-xl" />
  if (!board.cells.length) return null
  const { done, failed: lost, total } = board.progress
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Measurements
          <InfoTip>
            What each promoted model still lacks. CPU checks start at once; GPU trainings run one specialist at a time. Running jobs are never cancelled.
          </InfoTip>
        </CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">
          {done} of {total} done{lost > 0 && `, ${lost} failed`}
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Progress value={total ? (100 * (done + lost)) / total : 0} aria-label="Measured" />
        {board.error && (
          <Alert variant="destructive">
            <TriangleAlert />
            <AlertTitle>Measurements stopped advancing</AlertTitle>
            <AlertDescription>{board.error}</AlertDescription>
          </Alert>
        )}
        <Grid board={board} titles={titles} />
      </CardContent>
    </Card>
  )
}
