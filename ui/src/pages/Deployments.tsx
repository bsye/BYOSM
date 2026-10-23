import { Container, Rocket, Send, Square } from "lucide-react"
import { useEffect, useState } from "react"
import { api, post, runJob, type Summary } from "@/api"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { openJob } from "@/job-sheet"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { href } from "@/route"

const REFRESH_MS = 5000

type Deployment = { name: string; container: string; image: string; port: number | null; status: string; cpu?: string; memory?: string }
type Answer = { answer: { label?: string; probs?: Record<string, number> } & Record<string, unknown>; ms: number }

function useDeployments(): [Deployment[] | null, () => void] {
  const [rows, setRows] = useState<Deployment[] | null>(null)
  const load = () => void api<Deployment[]>("/deployments").then(setRows, () => setRows([]))
  useEffect(() => {
    load()
    const id = setInterval(load, REFRESH_MS)
    return () => clearInterval(id)
  }, [])
  return [rows, load]
}

function TryIt({ name }: { name: string }) {
  const [text, setText] = useState("")
  const [out, setOut] = useState<Answer | null>(null)
  const [error, setError] = useState("")
  const send = () => {
    setError("")
    post<Answer>(`/deployments/${encodeURIComponent(name)}/predict`, { input: text }).then(setOut, (e) => setError(String(e)))
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Test {name}
          <InfoTip>The request goes to the container, not to the app, so the answer and its time are the container's own.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2">
          <Input value={text} placeholder="an input to send" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && text && send()} />
          <Button disabled={!text} onClick={send}>
            <Send />
            Send
          </Button>
        </div>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {out && (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge>{String(out.answer.label ?? out.answer.action ?? "answer")}</Badge>
            {out.answer.call != null && (
              <span className="text-muted-foreground font-mono text-xs">
                {String(out.answer.tool)}({JSON.stringify(out.answer.call)})
              </span>
            )}
            {Object.entries(out.answer.probs ?? {})
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => (
                <span key={k} className="text-muted-foreground tabular-nums">
                  {k} {(v * 100).toFixed(1)}%
                </span>
              ))}
            <span className="text-muted-foreground ml-auto tabular-nums">{out.ms} ms</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function DeployPicker({ running }: { running: string[] }) {
  const [items, setItems] = useState<Summary[]>([])
  const [pick, setPick] = useState<string>("")
  useEffect(() => {
    api<Summary[]>("/specialists").then(setItems, () => setItems([]))
  }, [])
  const teams = items.filter((s) => s.kind === "team")
  const models = items.filter((s) => s.kind !== "team" && s.exported && !s.part_of)
  const start = () => runJob(["deploy", pick]).then((j) => openJob(j.id))
  const group = (label: string, rows: Summary[]) =>
    rows.length > 0 && (
      <SelectGroup>
        <SelectLabel>{label}</SelectLabel>
        {rows.map((s) => (
          <SelectItem key={s.name} value={s.name}>
            {s.title || s.name}
            {running.includes(s.name) ? " (running)" : ""}
          </SelectItem>
        ))}
      </SelectGroup>
    )
  return (
    <div className="flex items-center gap-2">
      <Select value={pick} onValueChange={setPick}>
        <SelectTrigger size="sm" className="w-64" aria-label="Specialist or team to deploy">
          <SelectValue placeholder="Specialist or team" />
        </SelectTrigger>
        <SelectContent>
          {group("Teams", teams)}
          {group("Specialists", models)}
        </SelectContent>
      </Select>
      <Button size="sm" disabled={!pick} onClick={start}>
        <Rocket />
        Deploy
      </Button>
    </div>
  )
}

export function Deployments() {
  const [rows, reload] = useDeployments()
  const [picked, setPicked] = useState<string | null>(null)
  if (rows == null) return <Skeleton className="h-64 rounded-xl" />
  const chosen = rows.find((r) => r.name === picked) ?? rows[0]
  const stop = (name: string) => post(`/deployments/${encodeURIComponent(name)}/stop`, {}).then(reload)
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold">Deployments</h1>
        <DeployPicker running={rows.map((r) => r.name)} />
      </div>
      {rows.length === 0 ? (
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Container />
            </EmptyMedia>
            <EmptyTitle>No model runs in a container</EmptyTitle>
            <EmptyDescription>Pick a specialist or a team above and deploy it.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Running containers
              <InfoTip>Each model runs alone: read-only filesystem, non-root user, 1 CPU and 1 GB at most, reachable only from this machine.</InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Endpoint</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">CPU</TableHead>
                  <TableHead className="text-right">Memory</TableHead>
                  <TableHead className="text-right">Stop</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.container} className="cursor-pointer" data-state={chosen?.name === r.name ? "selected" : undefined} onClick={() => setPicked(r.name)}>
                    <TableCell className="font-medium">
                      <a className="hover:underline" href={href("specialists", r.name)} onClick={(e) => e.stopPropagation()}>
                        {r.name}
                      </a>
                    </TableCell>
                    <TableCell className="text-muted-foreground font-mono text-xs">{r.port ? `127.0.0.1:${r.port}/v1/${r.name}/predict` : "-"}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{r.status}</Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground text-right tabular-nums">{r.cpu ?? "-"}</TableCell>
                    <TableCell className="text-muted-foreground text-right tabular-nums">{r.memory ?? "-"}</TableCell>
                    <TableCell className="text-right">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation()
                          void stop(r.name)
                        }}
                      >
                        <Square />
                        Stop
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      {chosen && <TryIt key={chosen.name} name={chosen.name} />}
    </div>
  )
}
