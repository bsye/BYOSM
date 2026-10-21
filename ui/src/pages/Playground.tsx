import { CircleCheck, CircleX, Loader2, Play, Swords } from "lucide-react"
import { useEffect, useState } from "react"
import { api, post, type Detail } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { InputGroup, InputGroupAddon, InputGroupTextarea } from "@/components/ui/input-group"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { rivalName } from "@/format"
import { GeneratedSuites } from "./GeneratedSuites"

type Prediction = { label?: string; probs?: Record<string, number>; entities?: { label: string; text: string }[] }
type Rival = { label?: string; confidence?: number; ms: number; error?: string }
export type Examples = { examples: Record<string, string[]>; rivals: string[]; repos: string[] }

const SHOWN = 5

export function useExamples(name: string): Examples | null {
  const [ex, setEx] = useState<Examples | null>(null)
  useEffect(() => {
    api<Examples>(`/specialists/${name}/examples`)
      .then(setEx)
      .catch(() => setEx(null))
  }, [name])
  return ex
}

function ExamplePicker({ ex, onPick }: { ex: Examples; onPick: (text: string) => void }) {
  const labels = Object.keys(ex.examples).sort()
  const [label, setLabel] = useState(labels[0] ?? "")
  if (!labels.length) return null
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Select value={label} onValueChange={setLabel}>
        <SelectTrigger size="sm" className="w-56" aria-label="Examples of a label">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {labels.map((l) => (
            <SelectItem key={l} value={l}>
              {l}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {(ex.examples[label] ?? []).map((t) => (
        <Button key={t} variant="outline" size="sm" className="text-muted-foreground max-w-xs" onClick={() => onPick(t)}>
          <span className="truncate">{t}</span>
        </Button>
      ))}
    </div>
  )
}

function Rivals({ ours, rivals }: { ours?: string; rivals: Record<string, Rival> | "loading" }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Rival</TableHead>
          <TableHead>Answer</TableHead>
          <TableHead>Confidence</TableHead>
          <TableHead className="text-right">Time</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rivals === "loading" ? (
          <TableRow>
            <TableCell colSpan={4} className="text-muted-foreground">
              <span className="flex items-center gap-2">
                <Loader2 className="size-4 animate-spin" />
                Asking every rival; the LLM can take a minute
              </span>
            </TableCell>
          </TableRow>
        ) : (
          Object.entries(rivals).map(([id, r]) => (
            <TableRow key={id}>
              <TableCell className="font-medium">{rivalName(id)}</TableCell>
              <TableCell>
                {r.error ? (
                  <span className="text-destructive text-xs">{r.error}</span>
                ) : (
                  <span className="flex items-center gap-1.5">
                    {r.label === ours ? <CircleCheck className="size-4" /> : <CircleX className="text-muted-foreground size-4" />}
                    {r.label}
                  </span>
                )}
              </TableCell>
              <TableCell className="text-foreground/70 tabular-nums">{r.confidence == null ? "-" : r.confidence.toFixed(3)}</TableCell>
              <TableCell className="text-foreground/70 text-right tabular-nums">
                {r.ms >= 1000 ? `${(r.ms / 1000).toFixed(1)} s` : `${Math.round(r.ms)} ms`}
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  )
}

function TryIt({ d }: { d: Detail }) {
  const ex = useExamples(d.name)
  const [text, setText] = useState("")
  const [out, setOut] = useState<Prediction | null>(null)
  const [ms, setMs] = useState<number | null>(null)
  const [rivals, setRivals] = useState<Record<string, Rival> | "loading" | null>(null)
  const [error, setError] = useState("")
  if (!d.exported) return <p className="text-muted-foreground text-sm">Export {d.name} first.</p>
  if (d.modality !== "text") return <p className="text-muted-foreground text-sm">Text only for now.</p>
  const run = (input = text) => {
    const t0 = performance.now()
    setRivals(null)
    post<Prediction>(`/specialists/${d.name}/predict`, { input })
      .then((p) => {
        setOut(p)
        setMs(performance.now() - t0)
        setError("")
      })
      .catch((e) => setError(String(e)))
  }
  const compare = () => {
    setRivals("loading")
    post<{ rivals: Record<string, Rival> }>(`/specialists/${d.name}/compare`, { input: text })
      .then((r) => setRivals(r.rivals))
      .catch((e) => setError(String(e)))
  }
  const top = Object.entries(out?.probs ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, SHOWN)
  return (
    <div className="flex flex-col gap-4">
      {ex && (
        <ExamplePicker
          ex={ex}
          onPick={(t) => {
            setText(t)
            run(t)
          }}
        />
      )}
      <InputGroup>
        <InputGroupTextarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Type an input" />
        <InputGroupAddon align="block-end" className="justify-end">
          <Button size="sm" disabled={!text.trim()} onClick={() => run()}>
            <Play />
            Run
          </Button>
        </InputGroupAddon>
      </InputGroup>
      {error && <p className="text-destructive text-sm">{error}</p>}
      {out && (
        <div className="grid gap-4 @3xl/main:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-3xl font-semibold">{out.label ?? "-"}</CardTitle>
              <CardAction className="text-muted-foreground text-sm tabular-nums">{ms == null ? "" : `${Math.round(ms)} ms`}</CardAction>
            </CardHeader>
            <CardContent className="space-y-2">
              {top.map(([label, p]) => (
                <div key={label} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className={label === out.label ? "font-medium" : "text-muted-foreground"}>{label}</span>
                    <span className="text-foreground/70 tabular-nums">{p.toFixed(3)}</span>
                  </div>
                  <Progress value={p * 100} />
                </div>
              ))}
              {out.entities?.map((e, i) => (
                <p key={i} className="text-sm">
                  <span className="font-medium">{e.label}</span>
                  <span className="text-muted-foreground"> {e.text}</span>
                </p>
              ))}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Rivals</CardTitle>
              <CardAction>
                <Button variant="outline" size="sm" disabled={rivals === "loading"} onClick={compare}>
                  <Swords />
                  Compare
                </Button>
              </CardAction>
            </CardHeader>
            <CardContent>{rivals ? <Rivals ours={out.label} rivals={rivals} /> : null}</CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

export function Playground({ d }: { d: Detail }) {
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <TryIt d={d} />
      <GeneratedSuites d={d} />
    </div>
  )
}
