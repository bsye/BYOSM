import { CircleCheck, CircleX, ScanSearch, TriangleAlert, type LucideIcon } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ErrorBar, XAxis, YAxis } from "recharts"
import { INTERVAL_BAR } from "@/chart-size"
import { runJob, type Detail, type Finding, type OverfitReport, type Severity } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { score } from "@/format"
import { openJob } from "@/job-sheet"

const SEVERITY: Record<Severity, { label: string; answer: string; icon: LucideIcon; variant: "outline" | "secondary" | "destructive" }> = {
  ok: { label: "OK", answer: "No", icon: CircleCheck, variant: "outline" },
  warn: { label: "Warning", answer: "Possibly", icon: TriangleAlert, variant: "secondary" },
  fail: { label: "Fail", answer: "Yes", icon: CircleX, variant: "destructive" },
}

const SPLITS = ["train", "dev", "calib", "test"] as const
const EXAMPLE_CHARS = 90

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
  const config = { score: { label: d.metric, color: "var(--chart-1)" } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="h-56 w-full">
      <BarChart accessibilityLayer data={rows} margin={{ left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="split" tickLine={false} tickMargin={10} axisLine={false} />
        <YAxis domain={[low, 1]} tickLine={false} axisLine={false} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
        <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
        <Bar dataKey="score" fill="var(--color-score)" radius={4}>
          <ErrorBar dataKey="ci" {...INTERVAL_BAR} />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

const TONE: Record<Severity, string> = { ok: "text-muted-foreground", warn: "text-foreground", fail: "text-destructive" }

function Row({ f, suiteName }: { f: Finding; suiteName: (s: string) => string }) {
  const s = SEVERITY[f.severity]
  return (
    <li className="flex items-center gap-3 py-2.5 text-sm">
      <s.icon className={`size-4 shrink-0 ${TONE[f.severity]}`} aria-label={s.label} />
      <span className="font-medium">{f.title}</span>
      <span className="text-muted-foreground min-w-0 truncate">{f.explanation}</span>
      <span className="ml-auto shrink-0">
        <InfoTip>
          <span className="block tabular-nums">{numbers(f)}</span>
          {f.suite && <span className="block">on {suiteName(f.suite)}</span>}
          {f.examples?.slice(0, 2).map((e, i) => (
            <span key={i} className="block">
              test "{clip(e.test)}", train "{clip(e.train)}"
            </span>
          ))}
        </InfoTip>
      </span>
    </li>
  )
}

export function Overfit({ d }: { d: Detail }) {
  const r = d.overfit
  const check = () => runJob(["overfit", d.name]).then((j) => openJob(j.id))
  const suiteName = (s: string) => (s === d.primary_suite ? d.benchmark : s)
  const stale = r && r.experiment !== d.promoted ? `, not ${d.promoted}` : ""
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Overfitting
          {r && (
            <span className="text-muted-foreground text-sm font-normal">
              {r.experiment}
              {stale}
            </span>
          )}
        </CardTitle>
        <CardAction>
          <Button variant="outline" size="sm" onClick={check}>
            <ScanSearch />
            {r ? "Check again" : "Check"}
          </Button>
        </CardAction>
      </CardHeader>
      {r ? (
        <CardContent className="space-y-6">
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-semibold">{SEVERITY[r.verdict].answer}</span>
            <span className="text-muted-foreground text-sm">overfitting</span>
          </div>
          <Scores d={d} r={r} />
          <ul className="divide-y border-y">
            {r.findings.map((f, i) => (
              <Row key={i} f={f} suiteName={suiteName} />
            ))}
          </ul>
          {r.skipped.map((s) => (
            <p key={s} className="text-muted-foreground/70 text-xs">
              Not checked: {s}
            </p>
          ))}
        </CardContent>
      ) : (
        <CardContent>
          <p className="text-muted-foreground text-sm">Not checked yet.</p>
        </CardContent>
      )}
    </Card>
  )
}
