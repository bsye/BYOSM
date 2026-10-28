import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import type { Search as SearchStats } from "@/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { InfoTip } from "@/components/info-tip"
import { StatCard, STAT_GRID } from "@/components/stat-card"
import { seconds } from "@/format"

const CHAR_PX = 7
const AXIS_PAD_PX = 12
const ROW_PX = 28

const config = {
  tries: { label: "Tries", color: "var(--chart-1)" },
  wins: { label: "Sure leads", color: "var(--chart-2)" },
} satisfies ChartConfig

export function Search({ s }: { s?: SearchStats }) {
  if (!s?.experiments) return null
  const axisWidth = Math.max(...s.families.map((f) => f.family.length)) * CHAR_PX + AXIS_PAD_PX
  const perLead = s.gpu_hours_per_sure_lead == null ? "-" : seconds(s.gpu_hours_per_sure_lead * 3600)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Experiment search
          <InfoTip>Each experiment belongs to one technique family. A sure lead is a promotion that beat the promoted model beyond seed noise.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={STAT_GRID}>
          <StatCard label="Families tried" value={String(s.families.length)} detail={`${s.experiments} experiments`} />
          <StatCard label="Sure leads" value={String(s.sure_leads)} />
          <StatCard label="GPU per sure lead" value={perLead} detail={`${seconds(s.gpu_hours * 3600)} in all`} />
          <StatCard label="Repeated edits" value={String(s.repeats)} />
        </div>
        <ChartContainer config={config} className="w-full" style={{ height: s.families.length * ROW_PX + 60 }}>
          <BarChart data={s.families} layout="vertical" margin={{ left: 8, right: 8 }} barGap={2}>
            <CartesianGrid horizontal={false} />
            <XAxis type="number" allowDecimals={false} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="family" tickLine={false} axisLine={false} width={axisWidth} interval={0} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Bar dataKey="tries" fill="var(--color-tries)" radius={4} />
            <Bar dataKey="wins" fill="var(--color-wins)" radius={4} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
