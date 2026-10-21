import { Gauge, Hammer, Lightbulb, ListFilter, Package, Save, Trash2, type LucideIcon, Wand2 } from "lucide-react"
import { useEffect, useState } from "react"
import { api, del, put, runJob, type Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { useLive } from "@/events"
import { metricName, ms, params, rivalName, score } from "@/format"
import { go } from "@/route"
import { BoardChart, boardRows } from "./BoardChart"
import { DataRows } from "./DataRows"
import { LabelChart } from "./LabelChart"
import { hasTeachers, Panel } from "./Panel"
import { Playground } from "./Playground"
import { Proposals } from "./Proposals"
import { Runs } from "./Runs"
import { SizeChart } from "./SizeChart"
import { Team, useTeam } from "./Team"
import { YourData } from "./YourData"

const ACTIONS: { label: string; icon: LucideIcon; args: (n: string) => string[] }[] = [
  { label: "Build", icon: Hammer, args: (n) => ["build", n] },
  { label: "Score", icon: Gauge, args: (n) => ["eval", n] },
  { label: "Improve", icon: Lightbulb, args: (n) => ["improve", n] },
  { label: "Export", icon: Package, args: (n) => ["export", n] },
]

const FOCUS_OTHERS = 5

const display = (d: Detail, model: string) => (d.baselines.includes(model) ? rivalName(model) : model)

function Board({ d }: { d: Detail }) {
  const [all, setAll] = useState(false)
  const every = boardRows(d.board, d.primary_suite)
  const keep = new Set([d.promoted, ...d.baselines])
  const others = every.filter((r) => !keep.has(r.model)).slice(0, FOCUS_OTHERS).map((r) => r.model)
  const rows = all ? every : every.filter((r) => keep.has(r.model) || others.includes(r.model))
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Test {metricName(d.metric)} on {d.benchmark}
        </CardTitle>
        <CardDescription>Each test score with its 95% interval, next to calib, the split used to choose.</CardDescription>
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
              <TableHead>Test [95% CI]</TableHead>
              <TableHead>Calib</TableHead>
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
                      <span className="font-medium" title={r.model}>
                        {display(d, r.model)}
                      </span>
                      {r.model === d.promoted && <Badge>Promoted</Badge>}
                      {d.baselines.includes(r.model) && <Badge variant="outline">Rival</Badge>}
                    </span>
                  </TableCell>
                  <TableCell className="tabular-nums">
                    <span className="font-medium">{score(r.test)}</span>{" "}
                    {res.ci && (
                      <span className="text-muted-foreground/70">
                        [{score(res.ci.lo)}, {score(res.ci.hi)}]
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{score(r.calib)}</TableCell>
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
        <CardDescription>Edited as JSON and validated by the server before it is saved.</CardDescription>
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
  if (team === undefined) return null
  if (team) return <Team view={team} />
  return <Model name={name} />
}

function Model({ name }: { name: string }) {
  const [d, setD] = useState<Detail | null>(null)
  const [error, setError] = useState("")
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  const load = () => api<Detail>(`/specialists/${name}`).then(setD).catch((e) => setError(String(e)))
  useEffect(() => {
    load()
  }, [name, finished])
  if (error) return <p className="text-destructive">{error}</p>
  if (!d) return null
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
          <h1 className="text-2xl font-semibold">{d.name}</h1>
          <p className="text-muted-foreground max-w-3xl text-sm">{d.description}</p>
          <p className="text-muted-foreground/70 text-xs">
            {d.modality} {d.task_type}, {d.labels} labels, {d.experiments.length} experiments
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {ACTIONS.map((a) => (
            <Button key={a.label} variant="outline" onClick={() => runJob(a.args(name)).then((j) => go("jobs", j.id))}>
              <a.icon />
              {a.label}
            </Button>
          ))}
          <Button onClick={() => runJob(["auto", name]).then((j) => go("jobs", j.id))}>
            <Wand2 />
            Autopilot
          </Button>
          <Button variant="destructive" size="icon" aria-label="Delete this specialist" title="Delete" onClick={remove}>
            <Trash2 />
          </Button>
        </div>
      </div>
      <Tabs defaultValue="board">
        <TabsList>
          <TabsTrigger value="board">Board</TabsTrigger>
          <TabsTrigger value="runs">Experiments</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
          {hasTeachers(d) && <TabsTrigger value="teachers">Teachers</TabsTrigger>}
          <TabsTrigger value="try">Try it</TabsTrigger>
          <TabsTrigger value="spec">Spec</TabsTrigger>
        </TabsList>
        <TabsContent value="board" className="space-y-4">
          <Board d={d} />
          <Proposals d={d} />
        </TabsContent>
        <TabsContent value="runs">
          <Runs d={d} />
        </TabsContent>
        <TabsContent value="data" className="space-y-4">
          {d.data && d.task_type === "classify" && <LabelChart data={d.data} />}
          <YourData name={name} />
          <DataRows name={name} />
        </TabsContent>
        <TabsContent value="teachers">{hasTeachers(d) && <Panel name={name} />}</TabsContent>
        <TabsContent value="try">
          <Playground d={d} />
        </TabsContent>
        <TabsContent value="spec">
          <SpecEditor d={d} onSaved={load} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
