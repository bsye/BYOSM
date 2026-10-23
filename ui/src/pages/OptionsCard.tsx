import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, ErrorBar, XAxis, YAxis } from "recharts"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"

type Share = { accuracy: number | null; ci: [number, number]; n: number }
type Scores = { exact: Share; tool_needed: Share; no_tool: Share }
type Report = { own?: Scores & { threshold: number }; compare?: { sample: number; model: Scores; rivals: Record<string, Scores> } }

const PARTS: [keyof Scores, string][] = [
  ["exact", "all requests"],
  ["tool_needed", "needs a tool"],
  ["no_tool", "needs none"],
]
const COLORS = 5

export function OptionsCard({ name }: { name: string }) {
  const [report, setReport] = useState<Report | null>(null)
  useEffect(() => {
    api<Report | null>(`/specialists/${name}/options`).then(setReport, () => setReport(null))
  }, [name])
  if (!report?.own && !report?.compare) return null
  const series: [string, Scores][] = report.compare ? [["this model", report.compare.model], ...Object.entries(report.compare.rivals)] : [["this model", report.own as Scores]]
  const keys = series.map((_, i) => `s${i}`)
  const config = Object.fromEntries(series.map(([who], i) => [keys[i], { label: who, color: `var(--chart-${(i % COLORS) + 1})` }])) satisfies ChartConfig
  const rows = PARTS.map(([part, label]) => ({
    part: label,
    ...Object.fromEntries(series.flatMap(([, s], i) => [[keys[i], s[part].accuracy], [`${keys[i]}ci`, s[part].accuracy == null ? [0, 0] : [s[part].accuracy - s[part].ci[0], s[part].ci[1] - s[part].accuracy]]])),
  }))
  const n = report.compare?.sample ?? report.own?.exact.n
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Choosing tools
          <InfoTip>A request counts as right only when exactly the tools it needs are chosen, or none when no offered tool fits. Bars show the 95% interval.</InfoTip>
        </CardTitle>
        <CardDescription>{n} held-out requests</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-64 w-full">
          <BarChart data={rows} margin={{ top: 8, left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="part" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} tickLine={false} axisLine={false} width={36} tickFormatter={(v) => Number(v).toFixed(1)} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            {keys.map((k) => (
              <Bar key={k} dataKey={k} fill={`var(--color-${k})`} radius={4}>
                <ErrorBar dataKey={`${k}ci`} width={4} strokeWidth={1} />
              </Bar>
            ))}
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
