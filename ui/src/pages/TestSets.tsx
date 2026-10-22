import { Check, Database, FileSpreadsheet, Loader2, Play, Sparkles, X } from "lucide-react"
import { useState } from "react"
import { download, runJob } from "@/api"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { ModelName, type ModelRef } from "@/components/model-name"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { rivalName } from "@/format"
import { openJob } from "@/job-sheet"
import { accuracy, useWorkbench, type TestSet } from "./workbench-api"

const OURS = "ours"
const DEFAULT_SET = "workbench"
const BENCHMARK_SET = "benchmark"
const DEFAULT_PER_LABEL = 10
const DEFAULT_SAMPLE = 20

const modelLabel = (m: string) => (m === OURS ? "this model" : rivalName(m))
const start = (args: string[], llm?: Record<string, unknown> | null) => runJob(args, llm ?? undefined).then((j) => openJob(j.id))

function Grid({ set, models, refs }: { set: TestSet; models: string[]; refs: Refs }) {
  return (
    <ScrollArea className="h-96">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Input</TableHead>
            <TableHead>Label</TableHead>
            {models.map((m) => {
              const a = accuracy(set, m)
              return (
                <TableHead key={m} className="text-right">
                  <ModelName model={refs.models[m === OURS ? (refs.promoted ?? "") : m]} shown={modelLabel(m)} />
                  <span className="text-muted-foreground ml-1 font-normal tabular-nums">{a.scored ? `${a.right}/${a.scored}` : "-"}</span>
                </TableHead>
              )
            })}
          </TableRow>
        </TableHeader>
        <TableBody>
          {[...set.rows].reverse().map((r) => (
            <TableRow key={r.id}>
              <TableCell className="max-w-md whitespace-normal">{r.input}</TableCell>
              <TableCell className="text-foreground/70">{r.label}</TableCell>
              {models.map((m) => {
                const got = set.results[m]?.[r.id]
                const right = got?.label === r.label
                return (
                  <TableCell key={m} className="text-right">
                    {got ? (
                      <span className={`inline-flex items-center gap-1 ${right ? "text-foreground/70" : "text-destructive"}`}>
                        {right ? <Check className="size-3.5" /> : <X className="size-3.5" />}
                        {right ? "" : got.label}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </ScrollArea>
  )
}

type Refs = { models: Record<string, ModelRef>; promoted: string | null }

export function TestSets({ name, labels, models, promoted }: { name: string; labels: number } & Refs) {
  const wb = useWorkbench(name)
  const llm = useLlmChoice()
  const [picked, setPicked] = useState<string | null>(null)
  const [perLabel, setPerLabel] = useState(DEFAULT_PER_LABEL)
  const [sample, setSample] = useState(DEFAULT_SAMPLE)
  const setName = picked ?? wb.sets[0]?.suite ?? DEFAULT_SET
  const set = wb.sets.find((s) => s.suite === setName)
  const chosen = wb.models?.chosen ?? []
  const unscored = set && !set.board ? Math.max(0, ...chosen.map((m) => set.rows.length - accuracy(set, m).scored)) : 0
  const fixed = Boolean(set?.board)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Test sets
          {wb.running > 0 && (
            <span className="text-muted-foreground flex items-center gap-1 text-xs font-normal">
              <Loader2 className="size-3.5 animate-spin" />
              {wb.running === 1 ? "1 job" : `${wb.running} jobs`}
            </span>
          )}
        </CardTitle>
        <CardAction className="flex gap-2">
          <Button size="sm" variant="outline" disabled={!set?.rows.length} onClick={() => download(`/specialists/${name}/written/${encodeURIComponent(setName)}.xlsx`, `${name}-${setName}-results.xlsx`)}>
            <FileSpreadsheet />
            Export
          </Button>
          <Select value={setName} onValueChange={setPicked}>
            <SelectTrigger size="sm" aria-label="Test set">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[...new Set([DEFAULT_SET, BENCHMARK_SET, ...wb.sets.map((s) => s.suite)])].map((s) => (
                <SelectItem key={s} value={s}>
                  {s} {wb.sets.find((x) => x.suite === s)?.rows.length ?? 0}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Input type="number" min={1} className="h-8 w-20 tabular-nums" aria-label="Inputs per label" value={perLabel} onChange={(e) => setPerLabel(Math.max(1, Number(e.target.value) || 1))} />
          <LlmPicker llm={llm} label="LLM that writes the inputs" />
          <Button size="sm" variant="outline" disabled={!llm.choice || fixed} onClick={() => start(["inputs", "write", name, "--per-label", String(perLabel), "--set", setName], llm.config())}>
            <Sparkles />
            Write {perLabel * labels}
          </Button>
          <Input type="number" min={1} className="ml-2 h-8 w-20 tabular-nums" aria-label="Rows from the benchmark" value={sample} onChange={(e) => setSample(Math.max(1, Number(e.target.value) || 1))} />
          <Button size="sm" variant="outline" disabled={fixed} onClick={() => start(["inputs", "sample", name, "--n", String(sample), "--set", setName])}>
            <Database />
            Add from benchmark
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ToggleGroup type="multiple" variant="outline" size="sm" value={chosen} onValueChange={(v) => v.length && wb.choose(v)}>
            {(wb.models?.available ?? []).map((m) => (
              <ToggleGroupItem key={m} value={m}>
                {modelLabel(m)}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <Button size="sm" className="ml-auto" disabled={!set?.rows.length || !unscored} onClick={() => start(["inputs", "run", name, "--set", setName])}>
            <Play />
            {fixed ? "Scored on the board" : unscored ? `Score ${unscored} rows` : "All scored"}
          </Button>
        </div>
        {set?.rows.length ? (
          <Grid set={set} models={chosen} refs={{ models, promoted }} />
        ) : (
          <p className="text-muted-foreground text-sm">No inputs in {setName} yet: write some or add rows from the benchmark.</p>
        )}
      </CardContent>
    </Card>
  )
}
