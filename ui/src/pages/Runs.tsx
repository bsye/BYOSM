import { useState } from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import type { Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { params, score } from "@/format"

const COLORS = 5
const DEFAULT_SHOWN = 4

function defaults(d: Detail): string[] {
  const latest = [...d.runs].reverse().map((r) => r.name)
  return [...new Set([d.promoted, ...latest].filter((x): x is string => !!x))].slice(0, DEFAULT_SHOWN)
}

export function Runs({ d }: { d: Detail }) {
  const [shown, setShown] = useState<string[]>(() => defaults(d))
  const runs = d.runs.filter((r) => shown.includes(r.name) && r.history.length)
  const epochs = Math.max(0, ...runs.map((r) => r.history.length))
  const data = Array.from({ length: epochs }, (_, i) => ({
    epoch: i + 1,
    ...Object.fromEntries(runs.map((r) => [r.name, r.history[i]?.dev])),
  }))
  const config = Object.fromEntries(
    runs.map((r, i) => [r.name, { label: r.name, color: `var(--chart-${(i % COLORS) + 1})` }]),
  ) satisfies ChartConfig
  const toggle = (name: string) =>
    setShown(shown.includes(name) ? shown.filter((x) => x !== name) : [...shown, name].slice(-COLORS))
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Training curves</CardTitle>
          <CardDescription>Dev {d.metric} after each epoch. Pick up to {COLORS} runs in the table below.</CardDescription>
        </CardHeader>
        <CardContent>
          {runs.length ? (
            <ChartContainer config={config} className="h-72 w-full">
              <LineChart data={data} margin={{ left: 8, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                <YAxis domain={["auto", "auto"]} tickLine={false} axisLine={false} width={44} tickFormatter={(v) => Number(v).toFixed(2)} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                {runs.map((r) => (
                  <Line key={r.name} dataKey={r.name} stroke={`var(--color-${r.name})`} dot={false} strokeWidth={2} />
                ))}
              </LineChart>
            </ChartContainer>
          ) : (
            <p className="text-muted-foreground text-sm">No training history recorded for the selected runs.</p>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Experiments</CardTitle>
          <CardDescription>Each run, what it was built from, and its test score on {d.primary_suite}.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>run</TableHead>
                <TableHead>built from</TableHead>
                <TableHead>test</TableHead>
                <TableHead>params</TableHead>
                <TableHead>train rows</TableHead>
                <TableHead>train time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...d.runs].reverse().map((r) => (
                <TableRow
                  key={r.name}
                  className="cursor-pointer"
                  data-state={shown.includes(r.name) ? "selected" : undefined}
                  onClick={() => toggle(r.name)}
                >
                  <TableCell className="font-medium">
                    {r.name} {r.name === d.promoted && <Badge className="ml-2">promoted</Badge>}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{r.parent ?? "-"}</TableCell>
                  <TableCell className="font-mono">{score(d.board[r.name]?.[d.primary_suite]?.objective)}</TableCell>
                  <TableCell>{params(r.params)}</TableCell>
                  <TableCell>{r.n_train ?? "-"}</TableCell>
                  <TableCell>{r.train_seconds == null ? "-" : `${Math.round(r.train_seconds / 6) / 10} min`}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
