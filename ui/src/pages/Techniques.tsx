import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { api } from "@/api"
import { axisWidth, ROW_PX } from "@/chart-size"
import { InfoTip } from "@/components/info-tip"
import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { seconds } from "@/format"
import { EffectChart } from "@/pages/EffectChart"
import { ProvenMethods } from "./ProvenMethods"
import { ScoutProposals } from "./ScoutProposals"
import { Catalogue, LAB, type Technique } from "./TechniqueCatalogue"

const FILTERS = ["all", "available", "built-in", "planned", "lab"] as const

const config = {
  tries: { label: "Tries", color: "var(--chart-1)" },
  wins: { label: "Sure leads", color: "var(--chart-2)" },
} satisfies ChartConfig

function Tried({ rows }: { rows: Technique[] }) {
  const tried = rows.filter((t) => t.tries).sort((a, b) => b.tries - a.tries)
  if (!tried.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Tries and sure leads per family</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="w-full" style={{ height: tried.length * ROW_PX + 60 }}>
          <BarChart data={tried} layout="vertical" margin={{ left: 8, right: 8 }} barGap={2}>
            <CartesianGrid horizontal={false} />
            <XAxis type="number" allowDecimals={false} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="title" tickLine={false} axisLine={false} width={axisWidth(tried.map((t) => t.title))} interval={0} />
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

export function Techniques() {
  const [rows, setRows] = useState<Technique[] | null>(null)
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all")
  useEffect(() => {
    api<Technique[]>("/techniques").then(setRows, () => setRows([]))
  }, [])
  if (!rows) return <Skeleton className="h-96 rounded-xl" />
  const shown = filter === "all" ? rows : rows.filter((t) => t.status === filter)
  const families = rows.filter((t) => t.status !== LAB)
  const tries = families.reduce((n, t) => n + t.tries, 0)
  const wins = families.reduce((n, t) => n + t.wins, 0)
  const hours = families.reduce((n, t) => n + t.gpu_hours, 0)
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="flex items-center gap-2 text-2xl font-semibold">
          Techniques
          <InfoTip>The families the proposer picks experiments from. Planned ones are not built yet; built-in ones run on every winning model.</InfoTip>
        </h1>
        <Tabs value={filter} onValueChange={(v) => setFilter(v as (typeof FILTERS)[number])}>
          <TabsList>
            {FILTERS.map((f) => (
              <TabsTrigger key={f} value={f} className="capitalize">
                {f}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>
      <div className={STAT_GRID}>
        <StatCard label="Available" value={String(families.filter((t) => t.status === "available").length)} detail={`${families.length} families in all`} />
        <StatCard label="Experiments" value={String(tries)} detail={`${families.filter((t) => t.tries).length} families tried`} />
        <StatCard label="Sure leads" value={String(wins)} detail={tries ? `${Math.round((100 * wins) / tries)}% of experiments` : undefined} />
        <StatCard
          label="GPU per sure lead"
          value={wins && hours ? seconds((hours / wins) * 3600) : "-"}
          detail={hours ? `${seconds(hours * 3600)} in all` : "not recorded yet"}
        />
      </div>
      <ScoutProposals />
      <ProvenMethods />
      <EffectChart families={families} />
      <Tried rows={families} />
      <Catalogue rows={shown} />
    </div>
  )
}
