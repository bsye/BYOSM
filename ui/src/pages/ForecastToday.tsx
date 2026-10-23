import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

type Band = { from: number; to: number; n: number }
type Reason = { feature: string; drop: number; value: string | number | null }
type Scored = { p: number; shown: Record<string, string | null>; features: Record<string, string | number | null>; why?: Reason[] }
export type Today = { year: number; trained_on: number[]; scored: number; expected: number; bands: Band[]; top: Scored[] }

const config = { n: { label: "entities", color: "var(--chart-1)" } } satisfies ChartConfig
const pct = (v: number) => `${Math.round(v * 100)}%`
const NEVER_DAYS = 9999
const value = (v: string | number | null) => (v == null ? "missing" : v === NEVER_DAYS ? "never" : typeof v === "number" ? String(Math.round(v * 10) / 10) : v)

type Check = { base_rate: number; predicted_rate?: number }

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <Card className="@container/card gap-1 py-4">
      <CardHeader className="px-4">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl font-semibold tabular-nums">{value}</CardTitle>
        {note && <p className="text-muted-foreground text-xs">{note}</p>}
      </CardHeader>
    </Card>
  )
}

function Bands({ bands }: { bands: Band[] }) {
  const rows = bands.map((b) => ({ band: `${pct(b.from)}–${pct(b.to)}`, n: b.n }))
  return (
    <ChartContainer config={config} className="h-48 w-full">
      <BarChart data={rows} margin={{ left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="band" tickLine={false} axisLine={false} interval={0} fontSize={11} />
        <YAxis scale="sqrt" tickLine={false} axisLine={false} width={48} tickFormatter={(v) => Number(v).toLocaleString()} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <Bar dataKey="n" fill="var(--color-n)" radius={4} />
      </BarChart>
    </ChartContainer>
  )
}

function Riskiest({ top }: { top: Scored[] }) {
  const { shown, footer } = useShowMore(top)
  const columns = Object.keys(top[0]?.shown ?? {})
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((c) => (
              <TableHead key={c}>{c}</TableHead>
            ))}
            <TableHead>Why</TableHead>
            <TableHead className="text-right">Chance</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {shown.map((r, i) => (
            <TableRow key={i}>
              {columns.map((c) => (
                <TableCell key={c} className="max-w-64 truncate">
                  {r.shown[c] ?? "—"}
                </TableCell>
              ))}
              <TableCell>
                <div className="flex flex-wrap gap-1">
                  {(r.why ?? []).slice(0, 2).map((w) => (
                    <Badge key={w.feature} variant="secondary" className="font-mono" title={`+${pct(w.drop)} because of this`}>
                      {w.feature} {value(w.value)}
                    </Badge>
                  ))}
                </div>
              </TableCell>
              <TableCell className="text-right tabular-nums">
                <Tooltip>
                  <TooltipTrigger>{pct(r.p)}</TooltipTrigger>
                  <TooltipContent className="max-w-xs font-mono text-xs">
                    {Object.entries(r.features)
                      .map(([k, v]) => `${k}: ${v ?? "—"}`)
                      .join(" · ")}
                  </TooltipContent>
                </Tooltip>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {footer}
    </>
  )
}

export function ForecastToday({ today, horizon, check, testYear }: { today: Today; horizon: number; check?: Check; testYear: number }) {
  const note = check?.predicted_rate != null ? `In ${testYear} it predicted ${(check.predicted_rate * 100).toFixed(1)}%, ${(check.base_rate * 100).toFixed(1)}% ended` : undefined
  return (
    <div className="grid gap-4">
      <div className="grid gap-4 @xl/main:grid-cols-3">
        <Stat label="Scored" value={today.scored.toLocaleString()} />
        <Stat label={`Expected to end within ${horizon === 1 ? "a year" : `${horizon} years`}`} value={today.expected.toLocaleString()} note={note} />
        <Stat label="Trained on" value={today.trained_on.join(", ")} />
      </div>
      <div className="grid gap-2">
        <h4 className="text-muted-foreground text-xs">Entities by chance of ending</h4>
        <Bands bands={today.bands} />
      </div>
      <div className="grid gap-2">
        <h4 className="text-muted-foreground text-xs">Most likely to end</h4>
        <Riskiest top={today.top} />
      </div>
    </div>
  )
}
