import { InfoTip } from "@/components/info-tip"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, XAxis } from "recharts"
import type { Summary } from "@/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"

const config = { lead: { label: "lead on test", color: "var(--chart-2)" } } satisfies ChartConfig

export function Overview({ items }: { items: Summary[] }) {
  const data = items
    .filter((s) => s.score != null && s.rival?.score != null)
    .map((s) => ({ name: s.name, lead: Number(((s.score ?? 0) - (s.rival?.score ?? 0)).toFixed(3)), rival: s.rival?.name }))
    .sort((a, b) => b.lead - a.lead)
  if (!data.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Lead over the best rival
          <InfoTip>Test score minus the strongest rival's, the rival chosen on calib.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-64 w-full">
          <BarChart accessibilityLayer data={data} margin={{ top: 24 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} tickMargin={10} axisLine={false} />
            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel={false} indicator="line" />} />
            <Bar dataKey="lead" radius={6}>
              <LabelList dataKey="lead" position="top" className="fill-foreground" fontSize={12} />
              {data.map((d) => (
                <Cell key={d.name} fill={d.lead >= 0 ? "var(--color-lead)" : "var(--destructive)"} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
