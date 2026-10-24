import { ShieldAlert } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ErrorBar, LabelList, Line, LineChart, XAxis, YAxis } from "recharts"
import { Badge } from "@/components/ui/badge"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { Leak, Score, Year } from "./forecast-api"

const config = { auc: { label: "AUC", color: "var(--chart-1)" } } satisfies ChartConfig
const LEAKS_SHOWN = 5
const MODELS = { gbm: "gradient boosting", logistic: "logistic regression", base: "base rate" }
const overTime = {
  gbm: { label: MODELS.gbm, color: "var(--chart-1)" },
  logistic: { label: MODELS.logistic, color: "var(--chart-2)" },
  base: { label: MODELS.base, color: "var(--chart-4)" },
} satisfies ChartConfig

export function OverTime({ years }: { years: Year[] }) {
  const rows = years.map((y) => ({ year: String(y.year), ...Object.fromEntries(Object.entries(MODELS).map(([k, m]) => [k, y.scores[m]?.auc])) }))
  return (
    <ChartContainer config={overTime} className="h-56 w-full">
      <LineChart data={rows} margin={{ top: 8, left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="year" tickLine={false} axisLine={false} />
        <YAxis domain={[0.4, 1]} tickLine={false} axisLine={false} width={36} tickFormatter={(v) => Number(v).toFixed(1)} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        {Object.keys(MODELS).map((k) => (
          <Line
            key={k}
            dataKey={k}
            type="monotone"
            stroke={`var(--color-${k})`}
            strokeWidth={2}
            dot={{ r: 3 }}
            strokeDasharray={k === "base" ? "4 4" : undefined}
          />
        ))}
      </LineChart>
    </ChartContainer>
  )
}

export function ScoreChart({ scores }: { scores: Record<string, Score> }) {
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

export function Leakage({ leaks, excluded, dropped }: { leaks: Leak[]; excluded: Record<string, string>; dropped: Record<string, string> }) {
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
