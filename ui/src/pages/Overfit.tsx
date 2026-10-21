import { CircleCheck, CircleX, ScanSearch, TriangleAlert, type LucideIcon } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ErrorBar, XAxis, YAxis } from "recharts"
import { runJob, type Detail, type Finding, type OverfitReport, type Severity } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { metricName, score } from "@/format"
import { go } from "@/route"

const SEVERITY: Record<Severity, { label: string; answer: string; icon: LucideIcon; variant: "outline" | "secondary" | "destructive" }> = {
  ok: { label: "OK", answer: "No", icon: CircleCheck, variant: "outline" },
  warn: { label: "Warning", answer: "Possibly", icon: TriangleAlert, variant: "secondary" },
  fail: { label: "Fail", answer: "Yes", icon: CircleX, variant: "destructive" },
}

const SPLITS = ["train", "dev", "calib", "test"] as const
const EXAMPLE_CHARS = 90

function SeverityBadge({ severity }: { severity: Severity }) {
  const s = SEVERITY[severity]
  return (
    <Badge variant={s.variant}>
      <s.icon />
      {s.label}
    </Badge>
  )
}

const number = (v: number) => (Number.isInteger(v) ? String(v) : score(v))

const numbers = (f: Finding) =>
  Object.entries(f.numbers)
    .filter(([, v]) => v != null)
    .map(([k, v]) => `${k.split("_").join(" ")} ${number(v as number)}`)
    .join(", ")

const clip = (text: string) => (text.length > EXAMPLE_CHARS ? `${text.slice(0, EXAMPLE_CHARS)}...` : text)

type Interval = { lo: number; hi: number } | null | undefined

function intervals(d: Detail, r: OverfitReport): Record<string, Interval> {
  const drift = r.findings.find((f) => f.check === "drift" && f.suite === r.suite)?.numbers
  const calib = drift ? { lo: drift.calib_lo as number, hi: drift.calib_hi as number } : null
  return { train: r.train.ci95, calib, test: d.board[r.experiment]?.[r.suite]?.ci }
}

function Scores({ d, r }: { d: Detail; r: OverfitReport }) {
  const cis = intervals(d, r)
  const rows = SPLITS.filter((s) => r.scores[s] != null).map((s) => {
    const v = r.scores[s] as number
    const b = cis[s]
    return { split: s, score: v, ci: b ? [v - b.lo, b.hi - v] : undefined }
  })
  const low = Math.max(0, Math.floor(Math.min(...rows.map((x) => x.score)) * 10) / 10 - 0.1)
  const config = { score: { label: metricName(r.metric), color: "var(--chart-1)" } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="h-56 w-full">
      <BarChart accessibilityLayer data={rows} margin={{ left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="split" tickLine={false} tickMargin={10} axisLine={false} />
        <YAxis domain={[low, 1]} tickLine={false} axisLine={false} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
        <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
        <Bar dataKey="score" fill="var(--color-score)" radius={4}>
          <ErrorBar dataKey="ci" width={4} strokeWidth={1} />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

function Row({ f, suiteName }: { f: Finding; suiteName: (s: string) => string }) {
  return (
    <div className="grid gap-2 py-3 md:grid-cols-[18rem_1fr] md:gap-4">
      <div className="flex flex-wrap items-center gap-2 self-start">
        <SeverityBadge severity={f.severity} />
        <span className="text-sm font-medium">{f.title}</span>
        {f.suite && <span className="text-muted-foreground text-sm">on {suiteName(f.suite)}</span>}
      </div>
      <div className="space-y-1">
        <p className="text-sm">{f.explanation}</p>
        <p className="text-muted-foreground text-xs tabular-nums">{numbers(f)}</p>
        {f.examples?.map((e, i) => (
          <p key={i} className="text-muted-foreground/70 text-xs break-words">
            test "{clip(e.test)}", train "{clip(e.train)}"
          </p>
        ))}
      </div>
    </div>
  )
}

export function Overfit({ d }: { d: Detail }) {
  const r = d.overfit
  const check = () => runJob(["overfit", d.name]).then((j) => go("jobs", j.id))
  const suiteName = (s: string) => (s === d.primary_suite ? d.benchmark : s)
  const stale = r && r.experiment !== d.promoted ? `, not the promoted ${d.promoted}` : ""
  return (
    <Card>
      <CardHeader>
        <CardTitle>Is it overfitting?</CardTitle>
        <CardDescription className="line-clamp-1">
          {r
            ? `Checked ${r.experiment}${stale}: ${metricName(r.metric)} on its own training rows against dev, calib and test.`
            : "Compares the score on training rows with dev, calib and test, and looks for test rows in training."}
        </CardDescription>
        <CardAction>
          <Button variant="outline" size="sm" onClick={check}>
            <ScanSearch />
            {r ? "Check again" : "Check for overfitting"}
          </Button>
        </CardAction>
      </CardHeader>
      {r ? (
        <CardContent className="space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-2xl font-semibold">{SEVERITY[r.verdict].answer}</span>
            <SeverityBadge severity={r.verdict} />
            <span className="text-muted-foreground text-sm">{r.summary}</span>
          </div>
          <Scores d={d} r={r} />
          <div className="divide-y border-y">
            {r.findings.map((f, i) => (
              <Row key={i} f={f} suiteName={suiteName} />
            ))}
          </div>
          {r.skipped.map((s) => (
            <p key={s} className="text-muted-foreground/70 text-xs">
              Not checked: {s}
            </p>
          ))}
        </CardContent>
      ) : (
        <CardContent>
          <p className="text-muted-foreground text-sm">Not checked yet. Run the check to see whether the promoted model memorised its data.</p>
        </CardContent>
      )}
    </Card>
  )
}
