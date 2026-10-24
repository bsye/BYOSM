import { Loader2, Play } from "lucide-react"
import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts"
import { api, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"

type Score = { mase: number; crps: number; train_s?: number; ms_per_forecast?: number; published?: boolean }
type Report = { domain: string; at: number | null; configs?: Record<string, Record<string, Score>>; overall?: Record<string, Score>; pending?: string[] }

const config = {
  mase: { label: "MASE", color: "var(--chart-1)" },
  crps: { label: "CRPS", color: "var(--chart-2)" },
} satisfies ChartConfig

function useSeries(): [Report[] | null, Set<string>] {
  const live = useLive()
  const jobs = Object.values(live.jobs).filter((j) => j.args[0] === "series")
  const running = new Set(jobs.filter((j) => j.status === "running" || j.status === "queued").map((j) => j.args[2]))
  const done = jobs.reduce((n, j) => n + (live.metrics[j.id] ?? []).filter((m) => m.kind === "series").length + (j.status === "done" ? 1 : 0), 0)
  const [rows, setRows] = useState<Report[] | null>(null)
  useEffect(() => {
    api<Report[]>("/series").then(setRows, () => setRows([]))
  }, [done])
  return [rows, running]
}

function DomainCard({ r, running }: { r: Report; running: boolean }) {
  const models = Object.entries(r.overall ?? {}).sort(([, a], [, b]) => a.mase - b.mase)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="capitalize">{r.domain}</CardTitle>
        <CardDescription>
          {r.configs ? Object.keys(r.configs).join(" · ") : "not run yet"}
          {r.pending?.length ? <span className="text-muted-foreground/60"> · {r.pending.join(" · ")}</span> : null}
        </CardDescription>
        <CardAction>
          <Button size="sm" variant="outline" disabled={running} onClick={() => void runJob(["series", "run", r.domain]).then((j) => openJob(j.id))}>
            {running ? <Loader2 className="animate-spin" /> : <Play />}
            {running ? "Running" : "Run"}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-4">
        {models.length ? (
          <>
            <ChartContainer config={config} className="h-48 w-full">
              <BarChart data={models.map(([name, s]) => ({ name, mase: s.mase, crps: s.crps }))} margin={{ top: 8, left: 8, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} fontSize={11} />
                <YAxis tickLine={false} axisLine={false} width={36} tickFormatter={(v) => Number(v).toFixed(1)} />
                <ReferenceLine y={1} strokeDasharray="4 4" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar dataKey="mase" fill="var(--color-mase)" radius={4} />
                <Bar dataKey="crps" fill="var(--color-crps)" radius={4} />
              </BarChart>
            </ChartContainer>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">MASE</TableHead>
                  <TableHead className="text-right">CRPS</TableHead>
                  <TableHead className="text-right">Train</TableHead>
                  <TableHead className="text-right">Per forecast</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map(([name, s]) => (
                  <TableRow key={name}>
                    <TableCell className="font-medium">
                      {name}
                      {s.published && (
                        <Badge variant="outline" className="ml-2">
                          published
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{s.mase.toFixed(3)}</TableCell>
                    <TableCell className="text-right tabular-nums">{s.crps.toFixed(3)}</TableCell>
                    <TableCell className="text-right tabular-nums">{s.train_s == null ? "-" : `${s.train_s.toFixed(1)} s`}</TableCell>
                    <TableCell className="text-right tabular-nums">{s.ms_per_forecast == null ? "-" : `${s.ms_per_forecast.toFixed(1)} ms`}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </>
        ) : running ? (
          <Skeleton className="h-48 rounded-lg" />
        ) : (
          <p className="text-muted-foreground text-sm">Run it to train a specialist and score it against the rivals.</p>
        )}
      </CardContent>
    </Card>
  )
}

export function TimeSeries() {
  const [rows, running] = useSeries()
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="flex items-center gap-2 text-2xl font-semibold">
        Time series
        <InfoTip>GIFT-Eval domains. MASE and CRPS are relative to seasonal naive (1.0, the dashed line): lower is better. Specialists train only on values before the test windows. Published scores come from the GIFT-Eval leaderboard and show only where our seasonal naive matches theirs.</InfoTip>
      </h1>
      {!rows ? (
        <Skeleton className="h-72 rounded-xl" />
      ) : !rows.length ? (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>No domains</EmptyTitle>
            <EmptyDescription>byosm series run energy</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <div className="grid auto-rows-fr gap-4 @5xl/main:grid-cols-2">
          {rows.map((r) => (
            <DomainCard key={r.domain} r={r} running={running.has(r.domain)} />
          ))}
        </div>
      )}
    </div>
  )
}
