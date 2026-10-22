import { Power, Square } from "lucide-react"
import { useState } from "react"
import { post } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { gigabytes } from "@/format"
import { modelName } from "@/jobs"
import type { Loaded } from "./models-api"

const ACTIVE = new Set(["running", "queued"])

const until = (iso: string | null) => (iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "-")

export function LocalModels({ items, onChange }: { items: Loaded[] | null; onChange: (items: Loaded[]) => void }) {
  const [error, setError] = useState("")
  const unload = (m: Loaded) =>
    post<Loaded[]>("/llm/local/unload", { server: m.server, model: m.model })
      .then(onChange)
      .catch((e) => setError(String(e)))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Loaded on this machine</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {error && <p className="text-destructive text-sm">{error}</p>}
        {items?.length ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Server</TableHead>
                <TableHead>Memory</TableHead>
                <TableHead>Unloads at</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((m) => (
                <TableRow key={`${m.server}-${m.model}`}>
                  <TableCell className="font-medium">
                    {m.model}
                    <span className="text-muted-foreground ml-2 text-xs">{[m.params, m.quantization].filter(Boolean).join(", ")}</span>
                  </TableCell>
                  <TableCell className="text-foreground/70">{m.server}</TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{gigabytes(m.bytes)}</TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{until(m.until)}</TableCell>
                  <TableCell className="text-right">
                    <Button size="sm" variant="outline" onClick={() => unload(m)}>
                      <Power />
                      Unload
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <p className="text-muted-foreground text-sm">{items ? "No local model is loaded." : "Checking Ollama and LM Studio."}</p>
        )}
      </CardContent>
    </Card>
  )
}

export function ModelsInUse() {
  const live = useLive()
  const [stopped, setStopped] = useState<Record<string, number>>({})
  const busy = new Map<string, Set<string>>()
  for (const [job, metrics] of Object.entries(live.metrics)) {
    if (!ACTIVE.has(live.jobs[job]?.status ?? "")) continue
    for (const m of metrics) if (m.kind === "llm") busy.set(String(m.model), new Set([...(busy.get(String(m.model)) ?? []), job]))
  }
  const stop = (model: string) => post<{ cancelled: string[] }>("/llm/stop", { model }).then((r) => setStopped((s) => ({ ...s, [model]: r.cancelled.length })))
  return (
    <Card>
      <CardHeader>
        <CardTitle>In use by running jobs</CardTitle>
      </CardHeader>
      <CardContent>
        {busy.size ? (
          <ul className="divide-y">
            {[...busy].map(([model, jobs]) => (
              <li key={model} className="flex items-center gap-3 py-2">
                <span className="font-medium">{modelName(model)}</span>
                <Badge variant="outline">{jobs.size === 1 ? "1 job" : `${jobs.size} jobs`}</Badge>
                {stopped[model] != null && <span className="text-muted-foreground text-xs">stopped {stopped[model]}</span>}
                <Button size="sm" variant="outline" className="ml-auto" onClick={() => stop(model)}>
                  <Square />
                  Stop its jobs
                </Button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground text-sm">No running job is calling an LLM.</p>
        )}
      </CardContent>
    </Card>
  )
}
