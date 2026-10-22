import { Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from "recharts"
import type { Detail, Suite } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { metricName, rivalName } from "@/format"
import { TestSets } from "./TestSets"

const GENERATED = "generated"

const config = {
  ours: { label: "ours", color: "var(--chart-1)" },
  rival: { label: "rival", color: "var(--chart-3)" },
} satisfies ChartConfig

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
      {d.task_type === "classify" && <TestSets name={d.name} labels={d.labels} />}
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
