import { Check, Database, FileSpreadsheet, Loader2, Play, Sparkles, X } from "lucide-react"
import { useMemo, useState } from "react"
import { download, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { LoadMore, useInfinite } from "@/components/infinite"
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
import { AddLlmRival } from "./AddLlmRival"
import { NewTestSet } from "./NewTestSet"
import { mergeResults, rowsPage, score, useWorkbench, type TestSet } from "./workbench-api"

const OURS = "ours"
const DEFAULT_SET = "workbench"
const BENCHMARK_SET = "benchmark"
const DEFAULT_PER_LABEL = 10
const DEFAULT_SAMPLE = 20

const modelLabel = (m: string) => (m === OURS ? "this model" : rivalName(m))
const start = (args: string[], llm?: Record<string, unknown> | null) => runJob(args, llm ?? undefined).then((j) => openJob(j.id))

function Grid({ name, set, models, refs, version }: { name: string; set: TestSet; models: string[]; refs: Refs; version: number }) {
  const fetchPage = useMemo(() => rowsPage(name, set.suite), [name, set.suite])
  const page = useInfinite(`${name}/${set.suite}`, fetchPage, mergeResults, version)
  const results = page.extra ?? {}
  return (
    <ScrollArea className="h-96">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Input</TableHead>
            <TableHead>Label</TableHead>
            {models.map((m) => {
              const a = score(set, m)
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
          {page.rows.map((r) => (
            <TableRow key={r.id}>
              <TableCell className="max-w-md whitespace-normal">{r.input}</TableCell>
              <TableCell className="text-foreground/70">{r.label}</TableCell>
              {models.map((m) => {
                const got = results[m]?.[r.id]
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
      <LoadMore onVisible={page.more} loading={page.loading} done={page.done} shown={page.rows.length} total={page.total} />
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
  const [instructions, setInstructions] = useState("")
  const [parallel, setParallel] = useState("")
  const parallelArgs = Number(parallel) > 0 ? ["--parallel", String(Math.floor(Number(parallel)))] : []
  const [created, setCreated] = useState<string[]>([])
  const setName = picked ?? wb.sets[0]?.suite ?? DEFAULT_SET
  const set = wb.sets.find((s) => s.suite === setName)
  const chosen = wb.models?.chosen ?? []
  const unscored = set && !set.board ? Math.max(0, ...chosen.map((m) => set.total - score(set, m).scored)) : 0
  const fixed = Boolean(set?.board)
  const names = [...new Set([DEFAULT_SET, BENCHMARK_SET, ...created, ...wb.sets.map((s) => s.suite)])]
  const writeArgs = [
    "inputs",
    "write",
    name,
    "--per-label",
    String(perLabel),
    "--set",
    setName,
    ...(instructions.trim() ? ["--instructions", instructions.trim()] : []),
    ...parallelArgs,
  ]
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
          <Button
            size="sm"
            variant="outline"
            disabled={!set?.total}
            onClick={() => download(`/specialists/${name}/written/${encodeURIComponent(setName)}.xlsx`, `${name}-${setName}-results.xlsx`)}
          >
            <FileSpreadsheet />
            Export
          </Button>
          <Select value={setName} onValueChange={setPicked}>
            <SelectTrigger size="sm" aria-label="Test set">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {names.map((s) => (
                <SelectItem key={s} value={s}>
                  {s} {(wb.sets.find((x) => x.suite === s)?.total ?? 0).toLocaleString()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <NewTestSet
            taken={names}
            onCreate={(n) => {
              setCreated([...created, n])
              setPicked(n)
            }}
          />
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Input
            type="number"
            min={1}
            className="h-8 w-20 tabular-nums"
            aria-label="Inputs per label"
            value={perLabel}
            onChange={(e) => setPerLabel(Math.max(1, Number(e.target.value) || 1))}
          />
          <LlmPicker llm={llm} label="LLM that writes the inputs" />
          <Input
            type="number"
            min={1}
            className="h-8 w-28 tabular-nums"
            placeholder="Parallel: auto"
            aria-label="LLM calls at the same time"
            value={parallel}
            onChange={(e) => setParallel(e.target.value)}
          />
          <Button size="sm" variant="outline" disabled={!llm.choice || fixed} onClick={() => start(writeArgs, llm.config())}>
            <Sparkles />
            Write {perLabel * labels}
          </Button>
          <Input
            type="number"
            min={1}
            className="ml-2 h-8 w-20 tabular-nums"
            aria-label="Rows from the benchmark"
            value={sample}
            onChange={(e) => setSample(Math.max(1, Number(e.target.value) || 1))}
          />
          <Button size="sm" variant="outline" disabled={fixed} onClick={() => start(["inputs", "sample", name, "--n", String(sample), "--set", setName])}>
            <Database />
            Add from benchmark
          </Button>
          {fixed && <InfoTip>This is the benchmark every model is compared on, so it stays as it is. Pick or create another set to add inputs.</InfoTip>}
        </div>
        <Input
          placeholder="Extra instructions for the writer, e.g. noisy phone calls with fillers"
          aria-label="Extra instructions"
          disabled={fixed}
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
        />
        <div className="flex flex-wrap items-center gap-2">
          <ToggleGroup type="multiple" variant="outline" size="sm" value={chosen} onValueChange={(v) => v.length && wb.choose(v)}>
            {(wb.models?.available ?? []).map((m) => (
              <ToggleGroupItem key={m} value={m}>
                {modelLabel(m)}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <AddLlmRival name={name} onAdded={wb.loadModels} />
          <Button
            size="sm"
            className="ml-auto"
            disabled={!set?.total || !unscored}
            onClick={() => start(["inputs", "run", name, "--set", setName, ...parallelArgs])}
          >
            <Play />
            {fixed ? "Scored on the board" : unscored ? `Score ${unscored} rows` : "All scored"}
          </Button>
        </div>
        {set?.total ? (
          <Grid name={name} set={set} models={chosen} refs={{ models, promoted }} version={wb.version} />
        ) : (
          <p className="text-muted-foreground text-sm">No inputs in {setName} yet: write some or add rows from the benchmark.</p>
        )}
      </CardContent>
    </Card>
  )
}
