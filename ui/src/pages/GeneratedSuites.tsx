import { Sparkles } from "lucide-react"
import { useState } from "react"
import { Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from "recharts"
import { runJob, type Detail, type Suite } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Input } from "@/components/ui/input"
import { metricName, rivalName } from "@/format"
import { WrittenInputs } from "./WrittenInputs"
import { openJob } from "@/job-sheet"

const SUITE_BASE = "llm-written"
const DEFAULT_TOTAL = 100
const GENERATED = "generated"

const config = {
  ours: { label: "ours", color: "var(--chart-1)" },
  rival: { label: "rival", color: "var(--chart-3)" },
} satisfies ChartConfig

const suiteName = (n: number) => (n === 1 ? SUITE_BASE : `${SUITE_BASE}-${n}`)

function freeName(d: Detail): string {
  const taken = new Set(d.suites.map((s) => s.name))
  let n = 1
  while (taken.has(suiteName(n))) n += 1
  return suiteName(n)
}

function GenerateCard({ d }: { d: Detail }) {
  const llm = useLlmChoice()
  const [total, setTotal] = useState(DEFAULT_TOTAL)
  const [error, setError] = useState("")
  const perLabel = Math.ceil(total / Math.max(1, d.labels))
  const start = () =>
    runJob(["testset", d.name, "--per-label", String(perLabel), "--suite", freeName(d)], llm.config())
      .then((j) => openJob(j.id))
      .catch((e) => setError(String(e)))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Write many inputs
          <InfoTip>The LLM writes new labelled inputs, then this model and every rival answer them.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Input
            type="number"
            min={1}
            className="h-8 w-24 tabular-nums"
            aria-label="How many inputs"
            value={total}
            onChange={(e) => setTotal(Math.max(1, Number(e.target.value) || 1))}
          />
          <LlmPicker llm={llm} label="LLM that writes the inputs" />
          <span className="text-muted-foreground text-xs tabular-nums">
            {perLabel} per label, {perLabel * d.labels} in all
          </span>
          <Button size="sm" className="ml-auto" disabled={!llm.choice || !d.promoted} onClick={start}>
            <Sparkles />
            Write and score
          </Button>
        </div>
        {error && <p className="text-destructive text-sm">{error}</p>}
      </CardContent>
    </Card>
  )
}

function SuiteCard({ d, suite }: { d: Detail; suite: Suite }) {
  const rows = [d.promoted, ...d.baselines]
    .filter((m): m is string => !!m && d.board[m]?.[suite.name]?.objective != null)
    .map((m) => ({
      model: d.baselines.includes(m) ? rivalName(m) : m,
      score: d.board[m][suite.name].objective as number,
      n: d.board[m][suite.name].n_test ?? 0,
      fill: d.baselines.includes(m) ? "var(--color-rival)" : "var(--color-ours)",
    }))
  const n = Math.max(0, ...rows.map((r) => r.n))
  const judged = suite.judged_by ? `, checked by ${suite.judged_by}` : ""
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="truncate">
            {n ? `${n} test inputs` : suite.name} written by {suite.written_by ?? "an LLM"}
          </span>
          <InfoTip>
            Test {metricName(d.metric)}{judged}. Written by an LLM, so not a public benchmark, and an LLM rival may agree
            with the LLM that wrote it.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="mt-auto">
        {rows.length ? (
          <ChartContainer config={config} className="h-48 w-full">
            <BarChart accessibilityLayer data={rows} margin={{ top: 20, left: 8, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="model" tickLine={false} tickMargin={10} axisLine={false} interval={0} />
              <YAxis hide domain={[0, 1]} />
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideIndicator />} />
              <Bar dataKey="score" radius={4} isAnimationActive={false}>
                <LabelList dataKey="score" position="top" className="fill-foreground tabular-nums" fontSize={12} formatter={(v) => Number(v).toFixed(3)} />
              </Bar>
            </BarChart>
          </ChartContainer>
        ) : (
          <p className="text-muted-foreground text-sm">Not scored yet. Press Score to score it.</p>
        )}
      </CardContent>
    </Card>
  )
}

export function GeneratedSuites({ d }: { d: Detail }) {
  return (
    <div className="flex flex-col gap-4">
      {d.task_type === "classify" && <GenerateCard d={d} />}
      <WrittenInputs name={d.name} />
      <div className="grid auto-rows-fr gap-4 @3xl/main:grid-cols-2">
        {d.suites
          .filter((s) => s.kind === GENERATED)
          .map((s) => (
            <SuiteCard key={s.name} d={d} suite={s} />
          ))}
      </div>
    </div>
  )
}
