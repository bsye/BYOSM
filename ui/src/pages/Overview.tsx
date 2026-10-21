import { Bar, BarChart, CartesianGrid, XAxis } from "recharts"
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
  return (
    <Card>
      <CardHeader>
        <CardTitle>Ours against the best rival</CardTitle>
        <CardDescription>Test score of each promoted specialist against its strongest rival, chosen on calib.</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-72 w-full">
          <BarChart accessibilityLayer data={data}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} tickMargin={10} axisLine={false} />
            <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="dashed" />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Bar dataKey="ours" fill="var(--color-ours)" radius={4} />
            <Bar dataKey="rival" fill="var(--color-rival)" radius={4} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
