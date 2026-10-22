import { useState } from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import type { Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { params, score, seconds } from "@/format"
import { Ablations } from "./Ablations"

const COLORS = 5
const DEFAULT_SHOWN = 4

const calib = (d: Detail, run: string) => d.board[run]?.[d.primary_suite]?.calib_objective ?? null

function byCalib(d: Detail): Detail["runs"] {
  const key = (name: string) => (name === d.promoted ? Infinity : (calib(d, name) ?? -Infinity))
  return d.runs.filter((r) => !r.ablation_arm).sort((a, b) => key(b.name) - key(a.name))
}

function defaults(d: Detail): string[] {
  return byCalib(d)
    .map((r) => r.name)
    .slice(0, DEFAULT_SHOWN)
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
          <CardTitle>Dev {d.metric} per epoch</CardTitle>
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
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run</TableHead>
                <TableHead>Built from</TableHead>
                <TableHead>Calib</TableHead>
                <TableHead>Test</TableHead>
                <TableHead>Params</TableHead>
                <TableHead>Train rows</TableHead>
                <TableHead>Train time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {byCalib(d).map((r) => (
                <TableRow
                  key={r.name}
                  className="cursor-pointer"
                  data-state={shown.includes(r.name) ? "selected" : undefined}
                  onClick={() => toggle(r.name)}
                >
                  <TableCell className="font-medium">
                    {r.name} {r.name === d.promoted && <Badge className="ml-2">Promoted</Badge>}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{r.parent ?? "-"}</TableCell>
                  <TableCell className="font-medium tabular-nums">{score(calib(d, r.name))}</TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{score(d.board[r.name]?.[d.primary_suite]?.objective)}</TableCell>
                  <TableCell>{params(r.params)}</TableCell>
                  <TableCell>{r.n_train ?? "-"}</TableCell>
                  <TableCell>{r.train_seconds == null ? "-" : seconds(r.train_seconds)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Ablations items={d.ablations} />
    </div>
  )
}
