import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Metric } from "@/events"
import { score } from "@/format"

const config = {
  dev_objective: { label: "dev score", color: "var(--chart-1)" },
  loss: { label: "loss", color: "var(--chart-2)" },
} satisfies ChartConfig

type Epoch = { epoch: number; epochs: number; loss: number; dev_objective: number; experiment: string }

function Epochs({ items }: { items: Epoch[] }) {
  const run = items.at(-1)
  const rows = items.filter((e) => e.experiment === run?.experiment)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Training {run?.experiment}</CardTitle>
        <CardDescription>
          Epoch {run?.epoch} of {run?.epochs}, dev score {score(run?.dev_objective)}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress value={run ? (run.epoch / run.epochs) * 100 : 0} />
        <ChartContainer config={config} className="h-56 w-full">
          <LineChart data={rows} margin={{ left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
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
              <span className="font-mono">{t.teacher}</span>
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
                <TableCell className="font-mono">{score(s.objective)}</TableCell>
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
  if (!epochs.length && !teachers.length && !scored.length) return null
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {epochs.length > 0 && <Epochs items={epochs} />}
      {teachers.length > 0 && <Teachers items={teachers} />}
      {scored.length > 0 && <Scores items={scored} />}
    </div>
  )
}
