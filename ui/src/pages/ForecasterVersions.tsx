import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { rivalName, score } from "@/format"
import type { ForecasterView, Version } from "./forecaster-api"

const DEFAULT_PRETRAINING = "v1"

const options = (v: Version) =>
  [
    (v.predict?.rates ?? []).some((r) => r !== 1) ? "multi-rate" : "",
    v.predict?.routing ? "routing" : "",
    v.fit?.quantile_boosting ? "quantile boosting" : "",
    v.fit?.pretraining && v.fit.pretraining !== DEFAULT_PRETRAINING ? `pretraining ${v.fit.pretraining}` : "",
    v.fit?.lab_method ? `lab ${v.fit.lab_method}` : "",
  ]
    .filter((o) => o)
    .join(", ") || "-"

const progress = {
  test: { label: "Test MASE", color: "var(--chart-1)" },
  calib: { label: "Calib MASE", color: "var(--chart-2)" },
} satisfies ChartConfig

function Progress({ f }: { f: ForecasterView }) {
  const versions = f.versions ?? []
  if (versions.length < 2) return null
  return (
    <ChartContainer config={progress} className="h-48 w-full">
      <LineChart data={versions.map((v) => ({ name: v.name, test: v.test, calib: v.calib }))} margin={{ top: 8, left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="name" tickLine={false} axisLine={false} />
        <YAxis tickLine={false} axisLine={false} width={40} domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(2)} />
        {f.rival?.score != null && f.rival.datasets === f.rival.of && (
          <ReferenceLine y={f.rival.score} strokeDasharray="4 4" label={{ value: rivalName(f.rival.name), position: "insideTopRight", fontSize: 11 }} />
        )}
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        <Line dataKey="test" stroke="var(--color-test)" strokeWidth={2} dot />
        <Line dataKey="calib" stroke="var(--color-calib)" strokeWidth={2} dot />
      </LineChart>
    </ChartContainer>
  )
}

export function Versions({ f }: { f: ForecasterView }) {
  return (
    <Card>
      <CardContent className="grid gap-4">
        <Progress f={f} />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Version</TableHead>
              <TableHead>Recipe</TableHead>
              <TableHead>Options</TableHead>
              <TableHead className="text-right">Training</TableHead>
              <TableHead className="text-right">Calib MASE</TableHead>
              <TableHead className="text-right">Test MASE</TableHead>
              <TableHead className="text-right">95% CI</TableHead>
              <TableHead className="text-right">Built</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[...(f.versions ?? [])].reverse().map((v) => (
              <TableRow key={v.name}>
                <TableCell className="font-medium">
                  {v.name}
                  {v.name === f.promoted && <Badge className="ml-2">promoted</Badge>}
                </TableCell>
                <TableCell>{v.recipe}</TableCell>
                <TableCell>{options(v)}</TableCell>
                <TableCell className="text-right tabular-nums">{2 ** (v.effort ?? 0)}x</TableCell>
                <TableCell className="text-right tabular-nums">{score(v.calib)}</TableCell>
                <TableCell className="text-right tabular-nums">{score(v.test)}</TableCell>
                <TableCell className="text-right tabular-nums">{v.ci ? `${score(v.ci[0])} to ${score(v.ci[1])}` : "-"}</TableCell>
                <TableCell className="text-right text-xs tabular-nums">
                  {new Date(v.at * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
