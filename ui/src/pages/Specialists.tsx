import { Hammer, Plus } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { useLive } from "@/events"
import { describe } from "@/jobs"
import type { NextStep } from "./NextSteps"
import { SpecialistCard } from "./SpecialistCard"
import { byDate, kindKey, kindTabs, type Group } from "./specialist-groups"
import { openJob } from "@/job-sheet"

const SKELETONS = 6
const EXAMPLE = "classify support emails into billing, shipping, technical or other"

function NewSpecialist({ children }: { children: React.ReactNode }) {
  const [prompt, setPrompt] = useState("")
  const [error, setError] = useState("")
  const start = () =>
    runJob(["auto", prompt])
      .then((job) => openJob(job.id))
      .catch((e) => setError(String(e)))
  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New specialist</DialogTitle>
          <DialogDescription>
            Describe the task in one sentence. The autopilot infers the labels, finds a benchmark and rivals, and trains until it wins or the budget ends.
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

const KIND_KEY = "byosm-specialists-kind"
const STAGE_KEY = "byosm-specialists-stage"
const ALL = "all"
const STAGES = { ready: "Ready", training: "In progress" }
type Stage = keyof typeof STAGES

const stageOf = (s: Summary): Stage => (s.ready === "ready" ? "ready" : "training")

function useRemembered<T extends string>(key: string, fallback: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      return (localStorage.getItem(key) as T | null) ?? fallback
    } catch {
      return fallback
    }
  })
  const set = (v: T) => {
    setValue(v)
    try {
      localStorage.setItem(key, v)
    } catch {
      return
    }
  }
  return [value, set]
}

function Section({ group, busy, showKind, steps }: { group: Group; busy: Map<string, string>; showKind: boolean; steps: Record<string, NextStep[]> }) {
  return (
    <section className="space-y-3">
      <h2 className="flex items-baseline gap-2 text-sm font-medium">
        {group.label}
        <span className="text-muted-foreground tabular-nums">{group.items.length}</span>
      </h2>
      <div className="grid auto-rows-fr gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
        {group.items.map((s) => (
          <SpecialistCard key={s.name} s={s} running={busy.get(s.name)} showKind={showKind} steps={steps[s.name]} />
        ))}
      </div>
    </section>
  )
}

export function Specialists() {
  const [items, setItems] = useState<Summary[] | null>(null)
  const [error, setError] = useState("")
  const [picked, setKind] = useRemembered<string>(KIND_KEY, ALL)
  const [remembered, setStage] = useRemembered<Stage>(STAGE_KEY, "ready")
  const stage: Stage = remembered in STAGES ? remembered : "ready"
  const everyone = (items ?? []).filter((s) => !s.part_of)
  const listed = everyone.filter((s) => stageOf(s) === stage)
  const tabs = kindTabs(listed)
  const kind = tabs.some((t) => t.key === picked) ? picked : ALL
  const shown = kind === ALL ? listed : listed.filter((s) => kindKey(s) === kind)
  const jobs = Object.values(useLive().jobs)
  const finished = jobs.filter((j) => j.status === "done").length
  const busy = new Map(jobs.filter((j) => j.status === "running" || j.status === "queued").map((j) => [describe(j).target ?? "", describe(j).title] as const))
  const [steps, setSteps] = useState<Record<string, NextStep[]>>({})
  useEffect(() => {
    api<Record<string, NextStep[]>>("/next-steps")
      .then(setSteps)
      .catch(() => setSteps({}))
  }, [finished])
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
      {items == null && !error && (
        <div className="grid auto-rows-fr gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
          {Array.from({ length: SKELETONS }, (_, i) => (
            <Skeleton key={i} className="h-44 rounded-xl" />
          ))}
        </div>
      )}
      {items && items.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Tabs value={kind} onValueChange={setKind}>
            <TabsList className="h-auto flex-wrap">
              <TabsTrigger value={ALL}>
                All <span className="text-muted-foreground tabular-nums">{listed.length}</span>
              </TabsTrigger>
              {tabs.map((t) => (
                <TabsTrigger key={t.key} value={t.key}>
                  {t.label} <span className="text-muted-foreground tabular-nums">{t.items.length}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          <ToggleGroup type="single" variant="outline" size="sm" value={stage} onValueChange={(v) => v && setStage(v as Stage)}>
            {(Object.keys(STAGES) as Stage[]).map((k) => (
              <ToggleGroupItem key={k} value={k}>
                {STAGES[k]} <span className="text-muted-foreground tabular-nums">{everyone.filter((s) => stageOf(s) === k).length}</span>
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
      )}
      {byDate(shown).map((g) => (
        <Section key={g.key} group={g} busy={busy} showKind={kind === ALL} steps={steps} />
      ))}
    </div>
  )
}
