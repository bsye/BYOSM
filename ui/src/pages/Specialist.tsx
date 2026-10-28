import { InfoTip } from "@/components/info-tip"
import { Forecaster, useForecaster } from "./Forecaster"
import { ForecastPage } from "./ForecastPage"
import { useForecasts } from "./forecast-api"
import { RunningHere } from "./RunningHere"
import { Archive, ArchiveRestore, Gauge, Hammer, Info, Lightbulb, ListFilter, Package, Save, Trash2, type LucideIcon, Wand2 } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, del, post, put, runJob, type Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { useLive } from "@/events"
import { signed, ms, params, rivalName, score } from "@/format"
import { go } from "@/route"
import { BoardChart, boardRows } from "./BoardChart"
import { DataRows } from "./DataRows"
import { LabelChart } from "./LabelChart"
import { DataSize } from "./DataSize"
import { Overfit } from "./Overfit"
import { hasTeachers, Panel } from "./Panel"
import { Playground, useExamples } from "./Playground"
import { Runs } from "./Runs"
import { SizeChart } from "./SizeChart"
import { PageSkeleton } from "@/components/page-skeleton"
import { ModelName } from "@/components/model-name"
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

const ACTIONS: { label: string; icon: LucideIcon; args: (n: string) => string[] }[] = [
  { label: "Build", icon: Hammer, args: (n) => ["build", n] },
  { label: "Score", icon: Gauge, args: (n) => ["eval", n] },
  { label: "Improve", icon: Lightbulb, args: (n) => ["improve", n] },
  { label: "Export", icon: Package, args: (n) => ["export", n] },
]

const FOCUS_OTHERS = 5

const display = (d: Detail, model: string) => (d.baselines.includes(model) ? rivalName(model) : model)

const LAB_MODALITIES = ["text", "audio", "image", "document"]
const canLab = (d: Detail) => LAB_MODALITIES.includes(d.modality)

function Board({ d }: { d: Detail }) {
  const [all, setAll] = useState(false)
  const every = boardRows(d.board, d.primary_suite)
  const keep = new Set([d.promoted, ...d.baselines])
  const others = every
    .filter((r) => !keep.has(r.model))
    .slice(0, FOCUS_OTHERS)
    .map((r) => r.model)
  const rows = all ? every : every.filter((r) => keep.has(r.model) || others.includes(r.model))
  const caughtLabels = Object.keys(d.board[d.promoted ?? ""]?.[d.primary_suite]?.caught ?? {})
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Test {d.metric} on {d.benchmark}
          <InfoTip>Test with its 95% interval, next to calib. Calib picks the model; test is only reported.</InfoTip>
        </CardTitle>
        {(every.length > rows.length || all) && (
          <CardAction>
            <Button variant="outline" size="sm" onClick={() => setAll(!all)}>
              <ListFilter />
              {all ? "Show the key models" : `Show all ${every.length}`}
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="space-y-6">
        {rows.length ? (
          <BoardChart rows={rows.map((r) => ({ ...r, model: display(d, r.model) }))} />
        ) : (
          <p className="text-muted-foreground text-sm">Nothing scored yet. Build or score this specialist to fill the board.</p>
        )}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>
                <span className="inline-flex items-center gap-1">
                  Test [95% CI]
                  <InfoTip>
                    The range the test score falls in when the test rows are resampled 1,000 times. With a bound like {d.metric}, a resample that misses the
                    bound scores 0, so a lower end of 0 means the model sits at the edge of the bound.
                  </InfoTip>
                </span>
              </TableHead>
              <TableHead>Calib</TableHead>
              {caughtLabels.map((l) => (
                <TableHead key={l}>Catches {l}</TableHead>
              ))}
              <TableHead>Params</TableHead>
              <TableHead>CPU per input</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const res = d.board[r.model][d.primary_suite]
              return (
                <TableRow key={r.model}>
                  <TableCell>
                    <span className="flex items-center gap-2">
                      <ModelName model={d.models[r.model]} shown={display(d, r.model)} />
                      {r.model === d.promoted && <Badge>Promoted</Badge>}
                      {d.baselines.includes(r.model) && <Badge variant="outline">Rival</Badge>}
                    </span>
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {res.unreached != null ? (
                      <Badge
                        variant="outline"
                        className="text-muted-foreground font-normal"
                        title="Never reaches the required precision on test, so it scores 0"
                      >
                        precision {(res.unreached * 100).toFixed(1)}%
                      </Badge>
                    ) : (
                      <span className="font-medium">{score(r.test)}</span>
                    )}{" "}
                    {res.unreached == null && res.ci && (
                      <span className="text-muted-foreground/70">
                        [{score(res.ci.lo)}, {score(res.ci.hi)}]
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{score(r.calib)}</TableCell>
                  {caughtLabels.map((l) => (
                    <TableCell key={l} className="text-foreground/70 tabular-nums">
                      {res.caught?.[l] != null ? `${((res.caught[l] as number) * 100).toFixed(1)}%` : "-"}
                    </TableCell>
                  ))}
                  <TableCell className="text-foreground/70 tabular-nums">{params(res.params)}</TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{ms(res.latency_ms)}</TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
        <SizeChart d={d} />
      </CardContent>
    </Card>
  )
}

function SpecEditor({ d, onSaved }: { d: Detail; onSaved: () => void }) {
  const [text, setText] = useState(JSON.stringify(d.spec, null, 2))
  const [note, setNote] = useState("")
  const save = async () => {
    try {
      await put(`/specialists/${d.name}/spec`, JSON.parse(text))
      setNote("saved")
      onSaved()
    } catch (e) {
      setNote(String(e))
    }
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>spec.yaml</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea className="h-[32rem] font-mono text-xs" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="flex items-center gap-3">
          <Button onClick={save}>
            <Save />
            Save changes
          </Button>
          <span className="text-muted-foreground text-sm">{note}</span>
        </div>
      </CardContent>
    </Card>
  )
}

export function Specialist({ name }: { name: string }) {
  const team = useTeam(name)
  const forecaster = useForecaster(name)
  const [risks] = useForecasts()
  if (team === undefined || forecaster === undefined || risks == null) return <PageSkeleton />
  if (team) return <Team view={team} />
  if (forecaster) return <Forecaster f={forecaster} />
  if (risks.some((r) => r.name === name)) return <ForecastPage name={name} />
  return <Model key={name} name={name} />
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

function Model({ name }: { name: string }) {
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
      <Tabs defaultValue="board">
        <TabsList>
          <TabsTrigger value="board">Board</TabsTrigger>
          <TabsTrigger value="runs">Experiments</TabsTrigger>
          {canLab(d) && <TabsTrigger value="lab">Lab</TabsTrigger>}
          <TabsTrigger value="data">Data</TabsTrigger>
          {hasTeachers(d) && <TabsTrigger value="teachers">Teachers</TabsTrigger>}
          <TabsTrigger value="try">Test</TabsTrigger>
          <TabsTrigger value="use">Use it</TabsTrigger>
          <TabsTrigger value="spec">Spec</TabsTrigger>
        </TabsList>
        <TabsContent value="board" className="space-y-4">
          <OptionsCard name={d.name} />
          <Board d={d} />
          <DataSize d={d} />
          <Overfit d={d} />
        </TabsContent>
        <TabsContent value="runs">
          <Runs d={d} />
        </TabsContent>
        <TabsContent value="lab">{canLab(d) && <Lab name={name} />}</TabsContent>
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
