import { Loader2, Play, ShieldAlert } from "lucide-react"
import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, ErrorBar, LabelList, XAxis, YAxis } from "recharts"
import { api, runJob } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import { NewForecast } from "./NewForecast"

type Score = { auc: number; auc_ci: [number, number]; ap: number; brier: number; base_rate: number }
type Leak = { feature: string; auc: number; suspect: boolean; why: string }
type Spec = { title: string; horizon_years: number; cutoffs: Record<string, number[]>; excluded: Record<string, string> }
type Result = { at: number; dropped?: Record<string, string>; splits: Record<string, { n: number; positive: number }>; leakage: Leak[]; scores: Record<string, Score> }
type Forecast = { name: string; spec: Spec; result: Result | null }

const config = { auc: { label: "AUC", color: "var(--chart-1)" } } satisfies ChartConfig
const LEAKS_SHOWN = 5

function useForecasts(): [Forecast[] | null, Set<string>, () => void] {
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "forecast")
  const running = new Set(jobs.filter((j) => j.status === "running" || j.status === "queued").map((j) => j.args[2]))
  const done = jobs.filter((j) => j.status === "done").length
  const [rows, setRows] = useState<Forecast[] | null>(null)
  const [saved, setSaved] = useState(0)
  useEffect(() => {
    api<Forecast[]>("/forecasts").then(setRows, () => setRows([]))
  }, [done, saved])
  return [rows, running, () => setSaved((n) => n + 1)]
}

function ScoreChart({ scores }: { scores: Record<string, Score> }) {
  const rows = Object.entries(scores)
    .map(([model, s]) => ({ model, auc: s.auc, ci: [s.auc - s.auc_ci[0], s.auc_ci[1] - s.auc] as [number, number], label: s.auc.toFixed(3) }))
    .sort((a, b) => b.auc - a.auc)
  return (
    <ChartContainer config={config} className="h-56 w-full">
      <BarChart data={rows} margin={{ top: 20, left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="model" tickLine={false} axisLine={false} interval={0} />
        <YAxis domain={[0.4, 1]} tickLine={false} axisLine={false} width={36} tickFormatter={(v) => Number(v).toFixed(1)} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <Bar dataKey="auc" fill="var(--color-auc)" radius={4}>
          <ErrorBar dataKey="ci" width={4} strokeWidth={1} />
          <LabelList dataKey="label" position="top" className="fill-muted-foreground" fontSize={11} />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

function Leakage({ leaks, excluded, dropped }: { leaks: Leak[]; excluded: Record<string, string>; dropped: Record<string, string> }) {
  return (
    <div className="grid gap-3">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Feature</TableHead>
            <TableHead className="text-right">AUC alone</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {leaks.slice(0, LEAKS_SHOWN).map((l) => (
            <TableRow key={l.feature}>
              <TableCell className="font-mono text-xs">
                {l.feature}
                {l.suspect && (
                  <Badge variant="destructive" className="ml-2" title={l.why}>
                    <ShieldAlert />
                    Leak, left out
                  </Badge>
                )}
              </TableCell>
              <TableCell className="text-right tabular-nums">{l.auc.toFixed(3)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="flex flex-wrap items-center gap-1.5">
        {(Object.keys(dropped).length > 0 || Object.keys(excluded).length > 0) && <span className="text-muted-foreground text-xs">Left out</span>}
        {[...Object.entries(dropped), ...Object.entries(excluded)].map(([col, why]) => (
          <Tooltip key={col}>
            <TooltipTrigger asChild>
              <Badge variant={col in dropped ? "destructive" : "outline"} className="font-mono">
                {col}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>{why}</TooltipContent>
          </Tooltip>
        ))}
      </div>
    </div>
  )
}

function ForecastCard({ f, running }: { f: Forecast; running: boolean }) {
  const test = f.result?.splits.test
  const facts = [`within ${f.spec.horizon_years} years`, `tested on ${f.spec.cutoffs.test.join(", ")}`, test ? `${test.n.toLocaleString()} rows, ${(test.positive * 100).toFixed(1)}% positive` : "not run yet"]
  return (
    <Card>
      <CardHeader>
        <CardTitle>{f.spec.title}</CardTitle>
        <CardDescription>{facts.join(" · ")}</CardDescription>
        <CardAction>
          <Button size="sm" variant="outline" disabled={running} onClick={() => void runJob(["forecast", "run", f.name]).then((j) => openJob(j.id))}>
            {running ? <Loader2 className="animate-spin" /> : <Play />}
            {running ? "Running" : "Run"}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {f.result ? (
          <div className="grid gap-6 @3xl/main:grid-cols-2">
            <div className="grid gap-2">
              <h3 className="text-sm font-medium">AUC on test, 95% CI</h3>
              <ScoreChart scores={f.result.scores} />
            </div>
            <div className="grid content-start gap-2">
              <h3 className="text-sm font-medium">Each feature alone</h3>
              <Leakage leaks={f.result.leakage} excluded={f.spec.excluded ?? {}} dropped={f.result.dropped ?? {}} />
            </div>
          </div>
        ) : (
          <Skeleton className="h-56 rounded-lg" />
        )}
      </CardContent>
    </Card>
  )
}

export function Forecasts() {
  const [rows, running, reload] = useForecasts()
  if (!rows) return <Skeleton className="h-72 rounded-xl" />
  return (
    <div className="grid gap-4">
      <div className="flex justify-end">
        <NewForecast onSaved={reload} />
      </div>
      {!rows.length && (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>No forecasts yet</EmptyTitle>
            <EmptyDescription>Keep a table with start and end dates, then create one.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      )}
      {rows.map((f) => (
        <ForecastCard key={f.name} f={f} running={running.has(f.name)} />
      ))}
    </div>
  )
}
