import { useEffect, useState } from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { useShowMore } from "@/components/show-more"
import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { compact } from "@/format"

type Compare = { model: string; per_item_usd: number; per_million_usd: number; break_even_items: number | null }
type Row = {
  name: string
  title: string
  build: { llm_usd: number; llm_calls: number; train_hours: number; compute_usd: number }
  evaluate_usd: number
  per_item: { measured: boolean; ours_usd: number; latency_ms: number | null; tokens_in?: number; tokens_out?: number }
  compare: Compare[]
}
type Costs = { settings: { watts: number; usd_per_kwh: number; compare: string[] }; specialists: Row[] }

const DEFAULT_RIVAL = "claude-sonnet-5-5"
const CURVE_POINTS = [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]
const tracked = (r: Row) => r.build.llm_calls > 0
const usd = (x: number) => (x >= 100 ? `$${Math.round(x).toLocaleString()}` : x >= 1 ? `$${x.toFixed(2)}` : x > 0 ? `$${x.toFixed(4)}` : "$0")

const config = {
  ours: { label: "Fine-tuned", color: "var(--chart-1)" },
  llm: { label: "LLM", color: "var(--chart-3)" },
} satisfies ChartConfig

function Curve({ row, rival }: { row: Row; rival: Compare }) {
  const built = row.build.llm_usd + row.build.compute_usd
  const data = CURVE_POINTS.map((n) => ({ n, ours: built + row.per_item.ours_usd * n, llm: rival.per_item_usd * n }))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Total cost by items served, {row.title}
          <InfoTip>Fine-tuned: what it cost to build, plus electricity per item. LLM: the measured tokens per item at {rival.model} list prices.</InfoTip>
        </CardTitle>
        <CardAction className="text-muted-foreground text-sm">
          {rival.break_even_items == null ? "never pays off" : `pays off after ${compact(rival.break_even_items)} items`}
        </CardAction>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-72 w-full">
          <LineChart data={data} margin={{ left: 8, right: 16 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="n" scale="log" type="number" domain={[1, 1e7]} ticks={CURVE_POINTS} tickFormatter={(v) => compact(Number(v))} tickLine={false} axisLine={false} />
            <YAxis scale="log" domain={["auto", "auto"]} tickFormatter={(v) => usd(Number(v))} tickLine={false} axisLine={false} width={64} allowDataOverflow />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => `${compact(Number(p?.[0]?.payload?.n))} items`} formatter={(v, k) => `${config[k as keyof typeof config].label} ${usd(Number(v))}`} />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line dataKey="ours" stroke="var(--color-ours)" strokeWidth={2} dot={false} />
            <Line dataKey="llm" stroke="var(--color-llm)" strokeWidth={2} dot={false} strokeDasharray="4 4" />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

function CostTable({ rows, rival, picked, pick }: { rows: Row[]; rival: string; picked: string | null; pick: (n: string) => void }) {
  const { shown, footer } = useShowMore(rows)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Built for, and when it pays off
          <InfoTip>LLM tokens are measured through Claude Code, which adds its own prompt to every call. That raises build cost and LLM cost per item alike, so the break-even holds better than either price. Models built before calls were recorded one by one count only the calls recorded since.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">LLM to build</TableHead>
              <TableHead className="text-right">Training</TableHead>
              <TableHead className="text-right">Per 1M items, LLM</TableHead>
              <TableHead className="text-right">Break-even</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((r) => {
              const c = tracked(r) ? r.compare.find((x) => x.model === rival) : undefined
              return (
                <TableRow key={r.name} className="cursor-pointer" data-state={picked === r.name ? "selected" : undefined} onClick={() => pick(r.name)}>
                  <TableCell className="font-medium">{r.title}</TableCell>
                  <TableCell className="text-right tabular-nums">{tracked(r) ? usd(r.build.llm_usd) : <span className="text-muted-foreground">not tracked</span>}</TableCell>
                  <TableCell className="text-muted-foreground text-right tabular-nums">
                    {r.build.train_hours.toFixed(1)} h, {usd(r.build.compute_usd)}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-right tabular-nums">{c ? usd(c.per_million_usd) : "not measured"}</TableCell>
                  <TableCell className="text-right tabular-nums">{c ? (c.break_even_items == null ? "never" : `${compact(c.break_even_items)} items`) : "-"}</TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
      {footer}
    </Card>
  )
}

export function CostsPage() {
  const [data, setData] = useState<Costs | null>(null)
  const [rival, setRival] = useState(DEFAULT_RIVAL)
  const [picked, setPicked] = useState<string | null>(null)
  useEffect(() => {
    api<Costs>("/costs").then(setData, () => setData({ settings: { watts: 0, usd_per_kwh: 0, compare: [] }, specialists: [] }))
  }, [])
  if (!data) return <Skeleton className="h-96 rounded-xl" />
  const rows = [...data.specialists].sort((a, b) => b.build.llm_usd - a.build.llm_usd)
  const measured = rows.filter((r) => tracked(r) && r.compare.some((c) => c.model === rival))
  const selected = rows.find((r) => r.name === picked) ?? measured[0]
  const pair = selected?.compare.find((c) => c.model === rival)
  const spent = rows.reduce((n, r) => n + r.build.llm_usd + r.build.compute_usd, 0)
  const evaluated = rows.reduce((n, r) => n + r.evaluate_usd, 0)
  const fastest = measured
    .map((r) => ({ r, n: r.compare.find((c) => c.model === rival)?.break_even_items ?? Infinity }))
    .sort((a, b) => a.n - b.n)[0]
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold">Costs</h1>
        <Select value={rival} onValueChange={setRival}>
          <SelectTrigger size="sm" className="w-72" aria-label="LLM to compare with">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {data.settings.compare.map((m) => (
              <SelectItem key={m} value={m}>
                Compared with {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className={STAT_GRID}>
        <StatCard label="Spent building" value={usd(spent)} detail="LLM at list prices, plus training power" />
        <StatCard label="Spent evaluating" value={usd(evaluated)} detail="rival answers and test sets" />
        <StatCard label="Pays off fastest" value={fastest && Number.isFinite(fastest.n) ? `${compact(fastest.n)} items` : "-"} detail={fastest ? fastest.r.title : "no LLM rival measured"} />
        <StatCard label="Power assumption" value={`${data.settings.watts} W`} detail={`at $${data.settings.usd_per_kwh} per kWh`} />
      </div>
      {selected && pair && <Curve row={selected} rival={pair} />}
      <CostTable rows={rows} rival={rival} picked={selected?.name ?? null} pick={setPicked} />
    </div>
  )
}
