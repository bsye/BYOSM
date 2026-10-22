import { Bar, BarChart, CartesianGrid, ErrorBar, LabelList, XAxis, YAxis } from "recharts"
import type { Result } from "@/api"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"

const config = {
  test: { label: "test", color: "var(--chart-1)" },
  calib: { label: "calib", color: "var(--chart-2)" },
} satisfies ChartConfig

type Row = { model: string; test: number; calib: number | null; ci: [number, number]; reach: string | null }

export function boardRows(board: Record<string, Record<string, Result>>, suite: string): Row[] {
  return Object.entries(board)
    .map(([model, by]) => ({ model, r: by[suite] }))
    .filter(({ r }) => r?.objective != null)
    .map(({ model, r }): Row => ({
      model,
      test: r.objective as number,
      calib: r.calib_objective,
      ci: r.ci ? ([(r.objective as number) - r.ci.lo, r.ci.hi - (r.objective as number)] as [number, number]) : [0, 0],
      reach: r.unreached != null ? `reaches ${(r.unreached * 100).toFixed(1)}%` : null,
    }))
    .sort((a, b) => b.test - a.test)
}

export function BoardChart({ rows }: { rows: Row[] }) {
  const low = Math.max(0, Math.floor(Math.min(...rows.map((r) => Math.min(r.test, r.calib ?? 1))) * 10) / 10 - 0.1)
  return (
    <ChartContainer config={config} className="h-80 w-full">
      <BarChart data={rows} margin={{ left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="model" tickLine={false} tickMargin={10} axisLine={false} interval={0} />
        <YAxis domain={[low, 1]} tickLine={false} axisLine={false} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        <Bar dataKey="test" fill="var(--color-test)" radius={4}>
          <ErrorBar dataKey="ci" width={4} strokeWidth={1} />
          <LabelList dataKey="reach" position="top" className="fill-muted-foreground" fontSize={11} formatter={(v) => (v == null ? "" : String(v))} />
        </Bar>
        <Bar dataKey="calib" fill="var(--color-calib)" radius={4} />
      </BarChart>
    </ChartContainer>
  )
}
