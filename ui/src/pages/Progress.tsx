import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts"
import type { Detail } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { rivalName, score } from "@/format"

const MIN_RUNS = 2

const config = {
  calib: { label: "calib", color: "var(--chart-1)" },
  test: { label: "test", color: "var(--chart-3)" },
} satisfies ChartConfig

export function Progress({ d }: { d: Detail }) {
  const prim = d.primary_suite
  const data = d.runs
    .filter((r) => !r.auxiliary && d.board[r.name]?.[prim])
    .sort((a, b) => (a.created ?? 0) - (b.created ?? 0))
    .map((r) => ({ name: r.name, calib: d.board[r.name][prim].calib_objective, test: d.board[r.name][prim].objective }))
  if (data.length < MIN_RUNS) return null
  const rival = d.rival?.score ? d.rival.score : null
  const best = Math.max(...data.map((x) => x.calib ?? 0))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Progress
          <InfoTip>
            Every experiment in the order it was built. Calib picks the model; test is only reported. The dashed line is the best rival on test.
          </InfoTip>
        </CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">
          best calib {score(best)}
          {rival != null && `, rival ${score(rival)}`}
        </CardAction>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-64 w-full">
          <LineChart data={data} margin={{ left: 8, right: 16, top: 20 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis domain={["auto", "auto"]} tickLine={false} axisLine={false} width={52} tickFormatter={(v) => Number(v).toFixed(3)} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            {rival != null && (
              <ReferenceLine
                y={rival}
                stroke="var(--muted-foreground)"
                strokeDasharray="4 4"
                label={{ value: rivalName(d.rival?.name), position: "insideTopLeft", fontSize: 11, fill: "var(--muted-foreground)" }}
              />
            )}
            {d.promoted && (
              <ReferenceLine
                x={d.promoted}
                stroke="var(--muted-foreground)"
                label={{ value: "promoted", position: "top", fontSize: 11, fill: "var(--muted-foreground)" }}
              />
            )}
            <Line dataKey="calib" stroke="var(--color-calib)" strokeWidth={2} dot={{ r: 3 }} connectNulls />
            <Line dataKey="test" stroke="var(--color-test)" strokeWidth={2} dot={{ r: 3 }} connectNulls />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
