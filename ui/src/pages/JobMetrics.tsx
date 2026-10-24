import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import { InfoTip } from "@/components/info-tip"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Metric } from "@/events"
import { score } from "@/format"

const config = {
  dev_objective: { label: "dev score, left axis", color: "var(--chart-1)" },
  loss: { label: "training loss, right axis", color: "var(--chart-2)" },
} satisfies ChartConfig

type Epoch = { epoch: number; epochs: number; loss: number; dev_objective: number; experiment: string }

function Epochs({ items }: { items: Epoch[] }) {
  const run = items.at(-1)
  const rows = items.filter((e) => e.experiment === run?.experiment)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Training {run?.experiment}
          <InfoTip>
            One point per epoch, a full pass over the training rows. The dev score is the specialist's metric on rows it does not train on: higher is better,
            and the best epoch is kept. The loss is the error on the training rows: lower is better. Dev falling while the loss keeps dropping is the sign of
            overfitting.
          </InfoTip>
        </CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">
          Epoch {run?.epoch} of {run?.epochs}, dev {score(run?.dev_objective)}
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress value={run ? (run.epoch / run.epochs) * 100 : 0} />
        <ChartContainer config={config} className="h-56 w-full">
          <LineChart data={rows} margin={{ left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="epoch" tickLine={false} axisLine={false} tickFormatter={(v) => `epoch ${v}`} />
            <YAxis yAxisId="dev" domain={["auto", "auto"]} tickLine={false} axisLine={false} width={44} />
            <YAxis yAxisId="loss" orientation="right" domain={["auto", "auto"]} tickLine={false} axisLine={false} width={44} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line yAxisId="dev" dataKey="dev_objective" stroke="var(--color-dev_objective)" strokeWidth={2} isAnimationActive={false} />
            <Line yAxisId="loss" dataKey="loss" stroke="var(--color-loss)" strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

type Teacher = { teacher: string; done: number; total: number }

function Teachers({ items }: { items: Teacher[] }) {
  const last = Object.values(Object.fromEntries(items.map((t) => [t.teacher, t])))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Teachers labelling</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {last.map((t) => (
          <div key={t.teacher} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span>{t.teacher}</span>
              <span>
                {t.done} / {t.total}
              </span>
            </div>
            <Progress value={(t.done / Math.max(1, t.total)) * 100} />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

type Writing = { split: string; done: number; total: number }

function Writes({ items }: { items: Writing[] }) {
  const last = Object.values(Object.fromEntries(items.map((w) => [w.split, w])))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Writing data</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {last.map((w) => (
          <div key={w.split} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span>{w.split}</span>
              <span className="text-foreground/70 tabular-nums">
                {w.done} / {w.total} calls
              </span>
            </div>
            <Progress value={(w.done / Math.max(1, w.total)) * 100} />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

type Scored = { predictor: string; predictor_kind: string; suite: string; objective: number | null }

function Scores({ items }: { items: Scored[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Scored</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>model</TableHead>
              <TableHead>suite</TableHead>
              <TableHead>test</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((s, i) => (
              <TableRow key={i}>
                <TableCell>
                  {s.predictor} <span className="text-muted-foreground text-xs">{s.predictor_kind}</span>
                </TableCell>
                <TableCell>{s.suite}</TableCell>
                <TableCell className="tabular-nums">{score(s.objective)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export function JobMetrics({ metrics }: { metrics: Metric[] }) {
  const of = <T,>(kind: string) => metrics.filter((m) => m.kind === kind) as unknown as T[]
  const epochs = of<Epoch>("epoch")
  const teachers = of<Teacher>("teacher")
  const scored = of<Scored>("scored")
  const writes = of<Writing>("writing")
  if (!epochs.length && !teachers.length && !scored.length && !writes.length) return null
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {epochs.length > 0 && <Epochs items={epochs} />}
      {writes.length > 0 && <Writes items={writes} />}
      {teachers.length > 0 && <Teachers items={teachers} />}
      {scored.length > 0 && <Scores items={scored} />}
    </div>
  )
}
