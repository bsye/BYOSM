import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import type { Summary } from "@/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"

const config = {
  ours: { label: "ours", color: "var(--chart-1)" },
  rival: { label: "best rival", color: "var(--chart-2)" },
} satisfies ChartConfig

export function Overview({ items }: { items: Summary[] }) {
  const data = items
    .filter((s) => s.score != null && s.rival?.score != null)
    .map((s) => ({ name: s.name, ours: s.score, rival: s.rival?.score, who: s.rival?.name }))
  if (!data.length) return null
  const wins = data.filter((d) => (d.ours ?? 0) > (d.rival ?? 0)).length
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Ahead on {wins} of {data.length} benchmarks
        </CardTitle>
        <CardDescription>Test score of each promoted specialist against its strongest rival, chosen on calib.</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-72 w-full">
          <BarChart data={data}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} tickLine={false} axisLine={false} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Bar dataKey="ours" fill="var(--color-ours)" radius={4} />
            <Bar dataKey="rival" fill="var(--color-rival)" radius={4} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
