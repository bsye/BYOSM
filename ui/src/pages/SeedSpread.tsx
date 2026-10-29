import { CartesianGrid, Scatter, ScatterChart, XAxis, YAxis, type ScatterShapeProps } from "recharts"
import type { Seeds } from "@/api"
import { axisWidth, ROW_PX } from "@/chart-size"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"
import { score } from "@/format"

const config = { run: { label: "seed", color: "var(--chart-2)" } } satisfies ChartConfig
const ROW_SCALE = 1.5
const AXIS_ROOM_PX = 40
const MEAN_HALF_PX = 9
const STEP = 0.01
const DOT_R = 6

type Point = { row: number; calib: number; title: string; detail: string }

const labelOf = (r: Seeds["recipes"][number]) => (r.promoted ? `${r.recipe} (promoted)` : r.recipe)

function spread(seeds: Seeds): { labels: string[]; runs: Point[]; means: Point[] } {
  const labels = seeds.recipes.map(labelOf)
  const runs = seeds.recipes.flatMap((r, row) =>
    r.runs.map((u) => ({ row, calib: u.calib, title: u.id, detail: `calib ${score(u.calib)}, test ${score(u.test)}` })),
  )
  const means = seeds.recipes.map((r, row) => ({ row, calib: r.mean, title: r.recipe, detail: `mean of ${r.runs.length} on calib ${score(r.mean)}` }))
  return { labels, runs, means }
}

function RunDot({ cx, cy }: ScatterShapeProps) {
  return <circle cx={cx} cy={cy} r={DOT_R} fill="var(--color-run)" stroke="var(--card)" strokeWidth={2} />
}

function MeanTick({ cx, cy }: ScatterShapeProps) {
  return <line x1={cx} x2={cx} y1={Number(cy) - MEAN_HALF_PX} y2={Number(cy) + MEAN_HALF_PX} stroke="var(--foreground)" strokeWidth={2} strokeLinecap="round" />
}

export function SeedSpread({ seeds }: { seeds: Seeds }) {
  if (!seeds.recipes.some((r) => r.runs.length > 1)) return null
  const { labels, runs, means } = spread(seeds)
  const values = runs.map((r) => r.calib)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Seed spread
          <InfoTip>
            Each dot is the same recipe trained on another seed, scored on {seeds.suite} calib; the line is their mean. A lead smaller than this spread is
            noise, so a promotion waits for a sure lead over every seed.
          </InfoTip>
        </CardTitle>
        {seeds.sd != null && (
          <CardAction>
            <Badge variant="outline" className="tabular-nums">
              seed sd {score(seeds.sd)}
            </Badge>
          </CardAction>
        )}
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="w-full" style={{ height: labels.length * ROW_PX * ROW_SCALE + AXIS_ROOM_PX }}>
          <ScatterChart margin={{ left: 8, right: 16 }}>
            <CartesianGrid horizontal={false} />
            <XAxis
              type="number"
              dataKey="calib"
              domain={[Math.floor(Math.min(...values) / STEP) * STEP, Math.ceil(Math.max(...values) / STEP) * STEP]}
              tickFormatter={(v) => score(Number(v))}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="number"
              dataKey="row"
              reversed
              domain={[-0.5, labels.length - 0.5]}
              ticks={labels.map((_, i) => i)}
              tickFormatter={(i) => labels[Number(i)] ?? ""}
              width={axisWidth(labels)}
              tickLine={false}
              axisLine={false}
            />
            <ChartTooltip
              cursor={false}
              content={({ payload }) => {
                const p = payload?.[0]?.payload as Point | undefined
                if (!p) return null
                return (
                  <div className="bg-background rounded-md border px-3 py-2 text-xs shadow-sm">
                    <div className="font-medium">{p.title}</div>
                    <div className="text-muted-foreground tabular-nums">{p.detail}</div>
                  </div>
                )
              }}
            />
            <Scatter data={runs} shape={RunDot} isAnimationActive={false} />
            <Scatter data={means} shape={MeanTick} isAnimationActive={false} />
          </ScatterChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
