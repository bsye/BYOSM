import { CartesianGrid, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from "recharts"
import type { Detail } from "@/api"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, type ChartConfig } from "@/components/ui/chart"
import { params, score } from "@/format"

const config = {
  ours: { label: "our experiments", color: "var(--chart-1)" },
  rivals: { label: "rivals", color: "var(--chart-2)" },
} satisfies ChartConfig

const PAD_X = 1.6
const PAD_Y = 0.03

type Point = { model: string; size: number; test: number }

export function SizeChart({ d }: { d: Detail }) {
  const points = Object.entries(d.board)
    .map(([model, by]) => ({ model, r: by[d.primary_suite] }))
    .filter(({ r }) => r?.objective != null && r.params)
    .map(({ model, r }) => ({ model, size: (r.params as number) / 1e6, test: r.objective as number }))
  const ours = points.filter((p) => !d.baselines.includes(p.model))
  const rivals = points.filter((p) => d.baselines.includes(p.model))
  if (!rivals.length || !ours.length) return null
  const sizes = points.map((p) => p.size)
  const tests = points.map((p) => p.test)
  const x: [number, number] = [Math.min(...sizes) / PAD_X, Math.max(...sizes) * PAD_X]
  const y: [number, number] = [Math.max(0, Math.min(...tests) - PAD_Y), Math.min(1, Math.max(...tests) + PAD_Y)]
  return (
    <div className="space-y-2">
    <p className="text-muted-foreground text-sm">Size vs test score</p>
    <ChartContainer config={config} className="h-72 w-full">
      <ScatterChart margin={{ left: 8, right: 16, bottom: 8 }}>
        <CartesianGrid />
        <XAxis type="number" dataKey="size" scale="log" domain={x} tickFormatter={(v) => `${Number(v).toPrecision(2)}M`} name="params" tickLine={false} axisLine={false} />
        <YAxis type="number" dataKey="test" domain={y} tickFormatter={(v) => Number(v).toFixed(2)} name="test" tickLine={false} axisLine={false} width={44} />
        <ZAxis range={[60, 60]} />
        <ChartTooltip
          content={({ payload }) => {
            const p = payload?.[0]?.payload as Point | undefined
            if (!p) return null
            return (
              <div className="bg-background rounded-md border px-3 py-2 text-xs shadow-sm">
                <div className="font-medium">{p.model}</div>
                <div className="text-muted-foreground">
                  {params(p.size * 1e6)}, test {score(p.test)}
                </div>
              </div>
            )
          }}
        />
        <ChartLegend content={<ChartLegendContent />} />
        <Scatter name="ours" data={ours} fill="var(--color-ours)" />
        <Scatter name="rivals" data={rivals} fill="var(--color-rivals)" />
      </ScatterChart>
    </ChartContainer>
    </div>
  )
}
