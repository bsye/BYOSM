import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import type { SplitSummary } from "@/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"

const SPLITS = ["train", "dev", "test"] as const
const MAX_LABELS = 40
const ROW_PX = 18

const config = {
  train: { label: "train", color: "var(--chart-1)" },
  dev: { label: "dev", color: "var(--chart-2)" },
  test: { label: "test", color: "var(--chart-3)" },
} satisfies ChartConfig

export function LabelChart({ data }: { data: Record<string, SplitSummary> }) {
  const labels = Object.entries(data.train?.labels ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_LABELS)
    .map(([label]) => label)
  const rows = labels.map((label) => ({ label, ...Object.fromEntries(SPLITS.map((s) => [s, data[s]?.labels[label] ?? 0])) }))
  const counts = SPLITS.map((s) => `${data[s]?.n ?? 0} ${s}`).join(", ")
  return (
    <Card>
      <CardHeader>
        <CardTitle>Labels</CardTitle>
        <CardDescription>
          {counts}. {labels.length < Object.keys(data.train?.labels ?? {}).length ? `The ${MAX_LABELS} largest labels shown.` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="w-full" style={{ height: Math.max(160, rows.length * ROW_PX + 60) }}>
          <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 8 }}>
            <CartesianGrid horizontal={false} />
            <XAxis type="number" tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="label" tickLine={false} axisLine={false} width={160} interval={0} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            {SPLITS.map((s) => (
              <Bar key={s} dataKey={s} stackId="n" fill={`var(--color-${s})`} />
            ))}
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
