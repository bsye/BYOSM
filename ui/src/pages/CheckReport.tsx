import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import type { CheckReport as Report } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

const MAX_LISTED = 30
const ROW_PX = 28

const VERDICT_BADGE = { ready: "default", "needs more": "secondary", fix: "destructive" } as const

const config = {
  rows: { label: "rows", color: "var(--chart-1)" },
  train_min: { label: "suggested minimum", color: "var(--chart-2)" },
} satisfies ChartConfig

function Coverage({ coverage }: { coverage: Report["coverage"] }) {
  const rows = Object.entries(coverage).map(([label, c]) => ({ label, rows: c.rows, train_min: c.train_min ?? 0 }))
  return (
    <ChartContainer config={config} className="w-full" style={{ height: Math.max(160, rows.length * ROW_PX + 60) }}>
      <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 8 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="label" tickLine={false} axisLine={false} width={160} interval={0} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        <Bar dataKey="rows" fill="var(--color-rows)" />
        <Bar dataKey="train_min" fill="var(--color-train_min)" />
      </BarChart>
    </ChartContainer>
  )
}

function Listed({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          {head.map((h) => (
            <TableHead key={h}>{h}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.slice(0, MAX_LISTED).map((r, i) => (
          <TableRow key={i}>
            {r.map((c, j) => (
              <TableCell key={j} className="max-w-md whitespace-normal">
                {c}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function CheckReport({ report }: { report: Report }) {
  const suspects = report.noise?.suspects ?? []
  const more = (n: number) => (n > MAX_LISTED ? ` The first ${MAX_LISTED} of ${n} shown.` : "")
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Data check <Badge variant={VERDICT_BADGE[report.verdict]}>{report.verdict}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {Object.keys(report.coverage).length > 0 && <Coverage coverage={report.coverage} />}
        {report.shortfalls.length > 0 && (
          <ul className="text-muted-foreground list-disc pl-5 text-sm">
            {report.shortfalls.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        )}
        {report.issues.length > 0 && (
          <div className="space-y-2">
            <h3 className="font-medium">Problems</h3>
            <p className="text-muted-foreground text-sm">
              {report.issues.length} issues.{more(report.issues.length)}
            </p>
            <Listed head={["row", "problem", "detail"]} rows={report.issues} />
          </div>
        )}
        <div className="space-y-2">
          <h3 className="font-medium">Labels</h3>
          <p className="text-muted-foreground text-sm">
            {report.noise
              ? `A probe agrees with ${Math.round(report.noise.agreement * 100)}% of the labels. Rows it confidently disagrees with are probably mislabeled.${more(suspects.length)}`
              : "No label probe for this table."}
          </p>
          {suspects.length > 0 && (
            <Listed
              head={["row", "labeled", "looks like", "reviewer"]}
              rows={suspects.map((s) => [s.row, s.given, s.probe, s.llm ? `${s.llm}: ${s.why ?? ""}` : ""])}
            />
          )}
        </div>
      </CardContent>
    </Card>
  )
}
