import { Forecaster } from "./Forecaster"
import { useForecaster } from "./forecaster-api"
import { ForecastPage } from "./ForecastPage"
import { useForecasts } from "./forecast-api"
import { RunningHere } from "./RunningHere"
import { Archive, ArchiveRestore, Gauge, Hammer, Info, Lightbulb, Package, Trash2, type LucideIcon, Wand2 } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, del, post, runJob, type Detail } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useLive } from "@/events"
import { signed, ms, params, rivalName, score } from "@/format"
import { go } from "@/route"
import { DataRows } from "./DataRows"
import { LabelChart } from "./LabelChart"
import { DataSize } from "./DataSize"
import { Overfit } from "./Overfit"
import { hasTeachers, Panel } from "./Panel"
import { Playground, useExamples } from "./Playground"
import { Runs } from "./Runs"
import { PageSkeleton } from "@/components/page-skeleton"
import { Health, type HealthReport } from "./Health"
import { titleOf } from "./specialist-groups"
import { NextSteps } from "./NextSteps"
import { PublishDialog } from "@/components/publish-dialog"
import { ContinueDialog } from "./ContinueDialog"
import { OptionsCard } from "./OptionsCard"
import { Team, useTeam } from "./Team"
import { UseIt } from "./UseIt"
import { YourData } from "./YourData"
import { openJob } from "@/job-sheet"
import { Lab } from "./Lab"
import { SpecEditor } from "./SpecEditor"
import { Board } from "./SpecialistBoard"
import { SuitesChart } from "./SuitesChart"

const ACTIONS: { label: string; icon: LucideIcon; args: (n: string) => string[] }[] = [
  { label: "Build", icon: Hammer, args: (n) => ["build", n] },
  { label: "Score", icon: Gauge, args: (n) => ["eval", n] },
  { label: "Improve", icon: Lightbulb, args: (n) => ["improve", n] },
  { label: "Export", icon: Package, args: (n) => ["export", n] },
]

export function Specialist({ name, tab }: { name: string; tab?: string }) {
  const team = useTeam(name)
  const forecaster = useForecaster(name)
  const [risks] = useForecasts()
  if (team === undefined || forecaster === undefined || risks == null) return <PageSkeleton />
  if (team) return <Team view={team} />
  if (forecaster) return <Forecaster f={forecaster} tab={tab} />
  if (risks.some((r) => r.name === name)) return <ForecastPage name={name} />
  return <Model key={name} name={name} tab={tab} />
}

function ModelStats({ d }: { d: Detail }) {
  const best = d.promoted ? d.board[d.promoted]?.[d.primary_suite] : undefined
  if (!best || d.score == null) return null
  const lead = d.rival?.score != null ? d.score - d.rival.score : null
  return (
    <div className={STAT_GRID}>
      <StatCard
        label={`Test ${d.metric}`}
        value={score(d.score)}
        detail={best.ci ? `95% CI ${score(best.ci.lo)} to ${score(best.ci.hi)}, ${d.benchmark}` : d.benchmark}
      />
      <StatCard label="Lead over best rival" value={signed(lead)} detail={d.rival ? `vs ${rivalName(d.rival.name)}` : "no rival scored yet"} />
      <StatCard label="CPU per input" value={ms(best.latency_ms)} detail={d.promoted ?? ""} />
      <StatCard label="Size" value={params(best.params)} detail="parameters, ONNX on CPU" />
    </div>
  )
}

function Model({ name, tab }: { name: string; tab?: string }) {
  const repo = useExamples(name)?.repos[0]
  const [d, setD] = useState<Detail | null>(null)
  const [error, setError] = useState("")
  const jobs = Object.values(useLive().jobs)
  const finished = jobs.filter((j) => j.status === "done").length
  const load = useCallback(
    () =>
      api<Detail>(`/specialists/${name}`)
        .then((x) => {
          setD(x)
          setError("")
        })
        .catch((e) => setError(String(e))),
    [name],
  )
  useEffect(() => {
    load()
  }, [load, finished])
  if (error) return <p className="text-destructive">{error}</p>
  if (!d) return <PageSkeleton />
  const shelve = () => post(`/specialists/${name}/archive`, { archived: !d.archived }).then(load)
  const remove = async () => {
    if (window.prompt(`Type ${name} to delete it and everything it built`) === name) {
      await del(`/specialists/${name}?confirm=${encodeURIComponent(name)}`)
      go()
    }
  }
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold">{titleOf(d)}</h1>
          <p className="text-muted-foreground max-w-3xl text-sm">{d.description}</p>
          {typeof d.spec.caveat === "string" && d.spec.caveat && (
            <Alert className="max-w-3xl">
              <Info />
              <AlertTitle>Built differently from the request</AlertTitle>
              <AlertDescription>{d.spec.caveat}</AlertDescription>
            </Alert>
          )}
          <p className="text-muted-foreground/70 text-xs">
            <span className="font-mono">{d.name}</span>, {d.modality} {d.task_type}, {d.labels} labels, {d.experiments.length} experiments
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {ACTIONS.map((a) => (
            <Button key={a.label} variant="outline" onClick={() => runJob(a.args(name)).then((j) => openJob(j.id))}>
              <a.icon />
              {a.label}
            </Button>
          ))}
          <ContinueDialog d={d} />
          <PublishDialog name={name} exported={d.exported} published={repo} />
          <Button onClick={() => runJob(["auto", name]).then((j) => openJob(j.id))}>
            <Wand2 />
            Autopilot
          </Button>
          <Button variant="outline" onClick={shelve}>
            {d.archived ? <ArchiveRestore /> : <Archive />}
            {d.archived ? "Restore" : "Archive"}
          </Button>
          <Button variant="destructive" size="icon" aria-label="Delete this specialist" title="Delete" onClick={remove}>
            <Trash2 />
          </Button>
        </div>
      </div>
      <RunningHere name={name} />
      <div className="grid gap-4 *:data-[slot=card]:h-full @3xl/main:grid-cols-2">
        <Health h={d.health as HealthReport | null} />
        <NextSteps d={d} />
      </div>
      <ModelStats d={d} />
      <Tabs defaultValue={tab ?? "board"}>
        <TabsList>
          <TabsTrigger value="board">Board</TabsTrigger>
          <TabsTrigger value="runs">Experiments</TabsTrigger>
          {d.lab && <TabsTrigger value="lab">Lab</TabsTrigger>}
          <TabsTrigger value="data">Data</TabsTrigger>
          {hasTeachers(d) && <TabsTrigger value="teachers">Teachers</TabsTrigger>}
          <TabsTrigger value="try">Test</TabsTrigger>
          <TabsTrigger value="use">Use it</TabsTrigger>
          <TabsTrigger value="spec">Spec</TabsTrigger>
        </TabsList>
        <TabsContent value="board" className="space-y-4">
          <OptionsCard name={d.name} />
          <Board d={d} />
          <SuitesChart d={d} />
          <DataSize d={d} />
          <Overfit d={d} />
        </TabsContent>
        <TabsContent value="runs">
          <Runs d={d} />
        </TabsContent>
        <TabsContent value="lab">{d.lab && <Lab name={name} kind="classifier" />}</TabsContent>
        <TabsContent value="data" className="space-y-4">
          {d.data && d.task_type === "classify" && <LabelChart data={d.data} />}
          <YourData name={name} />
          <DataRows name={name} />
        </TabsContent>
        <TabsContent value="teachers">{hasTeachers(d) && <Panel name={name} />}</TabsContent>
        <TabsContent value="try">
          <Playground d={d} />
        </TabsContent>
        <TabsContent value="use">
          <UseIt d={d} />
        </TabsContent>
        <TabsContent value="spec">
          <SpecEditor key={JSON.stringify(d.spec)} d={d} onSaved={load} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
