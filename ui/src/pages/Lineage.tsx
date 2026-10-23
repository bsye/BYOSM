import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import type { Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { score } from "@/format"
import { ContinueDialog } from "./ContinueDialog"

const config = {
  calib: { label: "calib", color: "var(--chart-1)" },
  test: { label: "test", color: "var(--chart-2)" },
} satisfies ChartConfig

export function chain(d: Detail, from: string | null): Detail["runs"] {
  const byName = new Map(d.runs.map((r) => [r.name, r]))
  const out: Detail["runs"] = []
  for (let r = from ? byName.get(from) : undefined; r && !out.includes(r); r = r.continued ? byName.get(r.continued) : undefined) out.unshift(r)
  return out
}

function delta(now: number | null | undefined, before: number | null | undefined) {
  if (now == null || before == null) return null
  const d = now - before
  return <span className={d >= 0 ? "text-foreground" : "text-destructive"}>{`${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)}`}</span>
}

export function Lineage({ d }: { d: Detail }) {
  const steps = chain(d, d.promoted ?? d.runs.at(-1)?.name ?? null)
  if (steps.length < 2) return null
  const res = (name: string) => d.board[name]?.[d.primary_suite]
  const rows = steps.map((r) => ({ name: r.name, calib: res(r.name)?.calib_objective ?? null, test: res(r.name)?.objective ?? null }))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Training history</CardTitle>
        <CardAction>
          <ContinueDialog d={d} />
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-6 @3xl/main:grid-cols-2">
        <ChartContainer config={config} className="h-56 w-full">
          <LineChart data={rows} margin={{ top: 8, left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} />
            <YAxis domain={["auto", "auto"]} tickLine={false} axisLine={false} width={44} tickFormatter={(v) => Number(v).toFixed(2)} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line dataKey="calib" stroke="var(--color-calib)" strokeWidth={2} dot={{ r: 3 }} />
            <Line dataKey="test" stroke="var(--color-test)" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ChartContainer>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Step</TableHead>
              <TableHead className="text-right">Rows</TableHead>
              <TableHead className="text-right">Calib</TableHead>
              <TableHead className="text-right">Change</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {steps.map((r, i) => (
              <TableRow key={r.name}>
                <TableCell className="font-medium">
                  {r.name}
                  {r.name === d.promoted && (
                    <Badge variant="secondary" className="ml-2">
                      promoted
                    </Badge>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums">{r.n_train?.toLocaleString() ?? "—"}</TableCell>
                <TableCell className="text-right tabular-nums">{score(rows[i].calib)}</TableCell>
                <TableCell className="text-right tabular-nums">{i ? delta(rows[i].calib, rows[i - 1].calib) : "start"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
