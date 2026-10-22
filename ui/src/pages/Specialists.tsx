import { Hammer, Loader2, Minus, Plus, TrendingDown, TrendingUp } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { useLive } from "@/events"
import { metricName, rivalName, score } from "@/format"
import { describe } from "@/jobs"
import { go, href } from "@/route"
import { Overview } from "./Overview"
import { SectionCards } from "./SectionCards"

const SKELETONS = 6
const EXAMPLE = "classify support emails into billing, shipping, technical or other"

function NewSpecialist({ children }: { children: React.ReactNode }) {
  const [prompt, setPrompt] = useState("")
  const [error, setError] = useState("")
  const start = () =>
    runJob(["auto", prompt])
      .then((job) => go("jobs", job.id))
      .catch((e) => setError(String(e)))
  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New specialist</DialogTitle>
          <DialogDescription>
            Describe the task in one sentence. The autopilot infers the labels, finds a benchmark and rivals, and trains
            until it wins or the budget ends.
          </DialogDescription>
        </DialogHeader>
        <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={EXAMPLE} />
        {error && <p className="text-destructive text-sm">{error}</p>}
        <DialogFooter>
          <Button disabled={!prompt.trim()} onClick={start}>
            <Hammer />
            Build it
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function Delta({ s }: { s: Summary }) {
  if (s.score == null || s.rival?.score == null) return <span className="text-muted-foreground">No rival scored yet</span>
  const d = s.score - s.rival.score
  const Icon = d > 0 ? TrendingUp : d < 0 ? TrendingDown : Minus
  return (
    <span className="text-muted-foreground flex w-full min-w-0 items-center gap-1">
      <Icon className="size-3.5 shrink-0" />
      <span className="text-foreground font-medium tabular-nums">
        {d > 0 ? "+" : ""}
        {d.toFixed(3)}
      </span>
      <span className="min-w-0 truncate">vs {rivalName(s.rival.name)}</span>
    </span>
  )
}

function SpecialistCard({ s, building }: { s: Summary; building: boolean }) {
  return (
    <a href={href("specialists", s.name)} className="h-full">
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{s.name}</CardTitle>
          <CardDescription className="line-clamp-1">{s.description}</CardDescription>
          <CardAction>
            {s.kind === "team" ? (
              <Badge variant="outline">Team of {s.members?.length}</Badge>
            ) : (
              s.exported && <Badge variant="outline">Exported</Badge>
            )}
          </CardAction>
        </CardHeader>
        <CardFooter className="mt-auto flex-col items-start gap-1">
          {s.score == null ? (
            <span className="text-muted-foreground flex items-center gap-2 text-sm">
              {building ? <Loader2 className="size-4 animate-spin" /> : null}
              {building ? "Building" : "Not built yet"}
            </span>
          ) : (
            <span className="text-3xl font-semibold tabular-nums" title={`Test ${metricName(s.metric)}, ${s.benchmark}`}>
              {score(s.score)}
            </span>
          )}
          {s.score != null && (
            <span className="w-full text-xs">
              <Delta s={s} />
            </span>
          )}
        </CardFooter>
      </Card>
    </a>
  )
}

export function Specialists() {
  const [items, setItems] = useState<Summary[] | null>(null)
  const [error, setError] = useState("")
  const jobs = Object.values(useLive().jobs)
  const finished = jobs.filter((j) => j.status === "done").length
  const busy = new Set(jobs.filter((j) => j.status === "running" || j.status === "queued").map((j) => describe(j).target ?? ""))
  useEffect(() => {
    api<Summary[]>("/specialists")
      .then(setItems)
      .catch((e) => setError(String(e)))
  }, [finished])
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Specialists</h1>
        </div>
        <NewSpecialist>
          <Button>
            <Plus />
            New specialist
          </Button>
        </NewSpecialist>
      </div>
      {error && <p className="text-destructive text-sm">Could not load the specialists: {error}</p>}
      {items && items.length > 0 && <SectionCards items={items.filter((s) => !s.part_of)} />}
      {items && items.length > 0 && <Overview items={items.filter((s) => !s.part_of)} />}
      {items && items.length === 0 && (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>No specialists yet</EmptyTitle>
            <EmptyDescription>Describe a task in one sentence and the autopilot builds the first one.</EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <NewSpecialist>
              <Button>
                <Plus />
                New specialist
              </Button>
            </NewSpecialist>
          </EmptyContent>
        </Empty>
      )}
      <div className="grid auto-rows-fr gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
        {items == null && !error && Array.from({ length: SKELETONS }, (_, i) => <Skeleton key={i} className="h-44 rounded-xl" />)}
        {(items ?? [])
          .filter((s) => !s.part_of)
          .sort((a, b) => Number(a.score == null) - Number(b.score == null))
          .map((s) => (
          <SpecialistCard key={s.name} s={s} building={busy.has(s.name)} />
        ))}
      </div>
    </div>
  )
}
