import { useEffect, useState } from "react"
import { api, del, put, runJob, type Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { useLive } from "@/events"
import { ms, params, score } from "@/format"
import { go } from "@/route"
import { BoardChart, boardRows } from "./BoardChart"
import { DataRows } from "./DataRows"
import { Playground } from "./Playground"

const ACTIONS: { label: string; args: (n: string) => string[] }[] = [
  { label: "Build", args: (n) => ["build", n] },
  { label: "Score", args: (n) => ["eval", n] },
  { label: "Improve", args: (n) => ["improve", n] },
  { label: "Export", args: (n) => ["export", n] },
  { label: "Autopilot", args: (n) => ["auto", n] },
]

const FOCUS_OTHERS = 5

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
          {d.metric} on {d.primary_suite}
        </CardTitle>
        <CardDescription>Test with its 95% interval and calib, the split that chooses. Test is only reported.</CardDescription>
        {every.length > rows.length || all ? (
          <Button variant="outline" size="sm" className="w-fit" onClick={() => setAll(!all)}>
            {all ? "Promoted, rivals and the best others" : `Show all ${every.length}`}
          </Button>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-6">
        {rows.length ? <BoardChart rows={rows} /> : <p className="text-muted-foreground text-sm">Nothing scored yet.</p>}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>model</TableHead>
              <TableHead>test [95% CI]</TableHead>
              <TableHead>calib</TableHead>
              <TableHead>params</TableHead>
              <TableHead>CPU</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const res = d.board[r.model][d.primary_suite]
              return (
                <TableRow key={r.model}>
                  <TableCell className="font-medium">
                    {r.model} {r.model === d.promoted && <Badge className="ml-2">promoted</Badge>}
                    {d.baselines.includes(r.model) && <Badge variant="outline" className="ml-2">rival</Badge>}
                  </TableCell>
                  <TableCell className="font-mono">
                    {score(r.test)} {res.ci && `[${score(res.ci.lo)}, ${score(res.ci.hi)}]`}
                  </TableCell>
                  <TableCell className="font-mono">{score(r.calib)}</TableCell>
                  <TableCell>{params(res.params)}</TableCell>
                  <TableCell>{ms(res.latency_ms)}</TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
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
          <Button onClick={save}>Save</Button>
          <span className="text-muted-foreground text-sm">{note}</span>
        </div>
      </CardContent>
    </Card>
  )
}

export function Specialist({ name }: { name: string }) {
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
        <div>
          <h1 className="text-2xl font-semibold">{d.name}</h1>
          <p className="text-muted-foreground max-w-3xl text-sm">{d.description}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge variant="secondary">{d.modality}</Badge>
            <Badge variant="secondary">{d.task_type}</Badge>
            <Badge variant="outline">{d.labels} labels</Badge>
            <Badge variant="outline">{d.experiments.length} experiments</Badge>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {ACTIONS.map((a) => (
            <Button key={a.label} variant="outline" onClick={() => runJob(a.args(name)).then((j) => go("jobs", j.id))}>
              {a.label}
            </Button>
          ))}
          <Button variant="destructive" onClick={remove}>
            Delete
          </Button>
        </div>
      </div>
      <Tabs defaultValue="board">
        <TabsList>
          <TabsTrigger value="board">Board</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="try">Try it</TabsTrigger>
          <TabsTrigger value="spec">Spec</TabsTrigger>
        </TabsList>
        <TabsContent value="board">
          <Board d={d} />
        </TabsContent>
        <TabsContent value="data">
          <DataRows name={name} />
        </TabsContent>
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
