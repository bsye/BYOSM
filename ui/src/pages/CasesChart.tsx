import { useState } from "react"
import { Bar, BarChart, CartesianGrid, ErrorBar, XAxis, YAxis } from "recharts"
import type { Detail } from "@/api"
import { axisWidth, INTERVAL_BAR } from "@/chart-size"
import { InfoTip } from "@/components/info-tip"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { rivalName, score } from "@/format"
import { caseSeries, caseSuites } from "./case-rows"
import { SERIES_TONES } from "./suite-rows"

const BAR_PX = 16
const AXIS_ROOM_PX = 72
const TICKS = [0, 0.25, 0.5, 0.75, 1]

export function CasesChart({ d }: { d: Detail }) {
  const suites = caseSuites(d)
  const [picked, setPicked] = useState(suites[0] ?? "")
  if (!suites.length) return null
  const suite = suites.includes(picked) ? picked : suites[0]
  const { models, rows } = caseSeries(d, suite)
  const config = Object.fromEntries(
    models.map(({ model, index }) => [`s${index}`, { label: d.baselines.includes(model) ? rivalName(model) : model, color: SERIES_TONES[index] }]),
  ) satisfies ChartConfig
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          By case
          <InfoTip>
            Exact accuracy on each kind of request in a public suite, with its 95% interval. A request is right only when exactly the tools that should be
            called are chosen, or none when none should be.
          </InfoTip>
        </CardTitle>
        {suites.length > 1 && (
          <CardAction>
            <Tabs value={suite} onValueChange={setPicked}>
              <TabsList>
                {suites.map((s) => (
                  <TabsTrigger key={s} value={s}>
                    {s}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </CardAction>
        )}
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="w-full" style={{ height: rows.length * models.length * BAR_PX + AXIS_ROOM_PX }}>
          <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }}>
            <CartesianGrid horizontal={false} />
            <XAxis type="number" domain={[0, 1]} ticks={TICKS} tickFormatter={(v) => Number(v).toFixed(2)} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="case" width={axisWidth(rows.map((r) => r.case))} tickLine={false} axisLine={false} interval={0} />
            <ChartTooltip content={<ChartTooltipContent formatter={(v, name) => `${config[String(name)]?.label ?? name} ${score(Number(v))}`} />} />
            <ChartLegend content={<ChartLegendContent />} />
            {models.map(({ index }) => (
              <Bar key={index} dataKey={`s${index}`} fill={`var(--color-s${index})`} radius={4} isAnimationActive={false}>
                <ErrorBar dataKey={`s${index}ci`} direction="x" {...INTERVAL_BAR} />
              </Bar>
            ))}
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
