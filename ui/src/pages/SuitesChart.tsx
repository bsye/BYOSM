import { Bar, BarChart, CartesianGrid, ErrorBar, XAxis, YAxis } from "recharts"
import { INTERVAL_BAR } from "@/chart-size"
import type { Detail } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { rivalName, score } from "@/format"
import { suiteSeries } from "./suite-rows"

const TONES = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"]
const FLOOR_STEP = 0.1
const WIDE_SPAN = 0.5
const WIDE_STEP = 0.2
const TICK_DIGITS = 10
const EPSILON = 1e-9

export function SuitesChart({ d }: { d: Detail }) {
  const { models, rows } = suiteSeries(d)
  if (!models.length) return null
  const config = Object.fromEntries(
    models.map((m, i) => [`s${i}`, { label: d.baselines.includes(m) ? rivalName(m) : m, color: TONES[i] }]),
  ) satisfies ChartConfig
  const values = rows.flatMap((r) => models.map((_, i) => r[`s${i}`])).filter((v): v is number => typeof v === "number")
  const low = Math.max(0, Math.floor(Math.min(...values) / FLOOR_STEP) * FLOOR_STEP - FLOOR_STEP)
  const step = 1 - low > WIDE_SPAN ? WIDE_STEP : FLOOR_STEP
  const ticks = Array.from({ length: Math.floor((1 - low) / step + EPSILON) + 1 }, (_, i) => Math.round((1 - i * step) * TICK_DIGITS) / TICK_DIGITS)
  const left = Object.keys(d.board).length - models.length
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Every suite
          <InfoTip>
            Test {d.metric} with its 95% interval on every test suite, for the promoted model, its rivals and the best other models scored on at least two
            suites. A missing bar is a suite that model was not scored on.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-72 w-full">
          <BarChart data={rows} margin={{ top: 12, left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="suite" tickLine={false} tickMargin={10} axisLine={false} interval={0} />
            <YAxis domain={[low, 1]} ticks={ticks} tickLine={false} axisLine={false} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
            <ChartTooltip content={<ChartTooltipContent formatter={(v, name) => `${config[String(name)]?.label ?? name} ${score(Number(v))}`} />} />
            <ChartLegend content={<ChartLegendContent />} />
            {models.map((_, i) => (
              <Bar key={i} dataKey={`s${i}`} fill={`var(--color-s${i})`} radius={4} isAnimationActive={false}>
                <ErrorBar dataKey={`s${i}ci`} {...INTERVAL_BAR} />
              </Bar>
            ))}
          </BarChart>
        </ChartContainer>
      </CardContent>
      {left > 0 && <CardFooter className="text-muted-foreground text-xs">{left} more models are on the board above</CardFooter>}
    </Card>
  )
}
