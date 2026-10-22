import { Gauge, Hammer, TrendingDown, TrendingUp } from "lucide-react"
import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts"
import { api, runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { useLive } from "@/events"
import { rivalName, score } from "@/format"
import { go, href } from "@/route"
import { SystemDiagram } from "./SystemDiagram"
import { TypeDemo } from "./TypeDemo"

type Split = { exact_frame: number; action_accuracy: number; arg_f1: number | null }
type Stream = {
  gate: { min_confidence: number; stable_for: number } | null
  test: { typed_share: number; early_share: number; wrong_per_100: number }
  test_at_end: { wrong_per_100: number }
  ms_per_keystroke: number
}
type Suite = { rows: { test: number }; rivals: Record<string, { test: Split; latency_ms: number | null }>; stream?: Stream }
export type TeamView = {
  name: string
  spec: {
    description: string
    nodes: Record<string, { specialist: string; role: string; task: string }>
    tools: Record<string, { when: string; args: Record<string, string>; call: string | null }>
  }
  board: Record<string, Suite> | null
  exported: boolean
}

const OURS = "system"
const config = {
  exact_frame: { label: "Exact", color: "var(--chart-1)" },
  action_accuracy: { label: "Action", color: "var(--chart-2)" },
  arg_f1: { label: "Arguments", color: "var(--chart-3)" },
} satisfies ChartConfig

const pct = (x: number) => `${Math.round(x * 100)}%`

function Stat({ label, value, detail }: { label: string; value: string; detail?: React.ReactNode }) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl font-semibold tabular-nums">{value}</CardTitle>
      </CardHeader>
      {detail && <CardFooter className="text-muted-foreground mt-auto text-xs">{detail}</CardFooter>}
    </Card>
  )
}

function Stats({ suite }: { suite: Suite }) {
  const ours = suite.rivals[OURS]?.test
  const llm = Object.entries(suite.rivals).find(([k]) => k !== OURS)
  const lead = ours && llm ? ours.exact_frame - llm[1].test.exact_frame : null
  const st = suite.stream
  return (
    <div className="grid auto-rows-fr gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
      <Stat
        label="Exact"
        value={score(ours?.exact_frame)}
        detail={
          lead != null && (
            <span className="flex items-center gap-1">
              {lead >= 0 ? <TrendingUp className="size-3.5" /> : <TrendingDown className="size-3.5" />}
              <span className="text-foreground font-medium tabular-nums">
                {lead >= 0 ? "+" : ""}
                {lead.toFixed(3)}
              </span>
              vs {rivalName(llm?.[0])}
            </span>
          )
        }
      />
      {st?.gate && <Stat label="Typed before firing" value={pct(st.test.typed_share)} detail={`${pct(st.test.early_share)} fire early`} />}
      {st?.gate && <Stat label="Wrong per 100" value={st.test.wrong_per_100.toFixed(1)} detail={`${st.test_at_end.wrong_per_100.toFixed(1)} at the end`} />}
      {st && <Stat label="Per keystroke" value={`${st.ms_per_keystroke.toFixed(1)} ms`} detail="CPU" />}
    </div>
  )
}

function Results({ name, suite }: { name: string; suite: Suite }) {
  const data = Object.entries(suite.rivals).map(([rival, v]) => ({ rival: rival === OURS ? "this team" : rivalName(rival), ...v.test }))
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Test on {name}, {suite.rows.test} requests
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-64 w-full">
          <BarChart accessibilityLayer data={data}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="rival" tickLine={false} tickMargin={10} axisLine={false} />
            <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="dashed" />} />
            <ChartLegend content={<ChartLegendContent />} />
            {Object.keys(config).map((k) => (
              <Bar key={k} dataKey={k} fill={`var(--color-${k})`} radius={6} />
            ))}
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

export function Team({ view }: { view: TeamView }) {
  const [suiteName, suite] = Object.entries(view.board ?? {})[0] ?? []
  const names = Object.values(view.spec.nodes).map((n) => n.specialist)
  const start = (args: string[]) => runJob(args).then((j) => go("jobs", j.id))
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold">{view.name}</h1>
          <p className="text-muted-foreground text-sm">
            Team of {names.length}:{" "}
            {names.map((n, i) => (
              <span key={n}>
                {i > 0 && ", "}
                <a className="hover:text-foreground underline underline-offset-4" href={href("specialists", n)}>
                  {n}
                </a>
              </span>
            ))}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => start(["system", "eval", view.name])}>
            <Gauge />
            Score
          </Button>
          <Button onClick={() => start(["system", "build", view.name])}>
            <Hammer />
            Build
          </Button>
        </div>
      </div>
      {suite && <Stats suite={suite} />}
      {view.exported && <TypeDemo system={view.name} />}
      <SystemDiagram nodes={view.spec.nodes} tools={view.spec.tools ?? {}} />
      {suite && suiteName && <Results name={suiteName} suite={suite} />}
    </div>
  )
}

export function useTeam(name: string): TeamView | null | undefined {
  const [view, setView] = useState<TeamView | null | undefined>(undefined)
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  useEffect(() => {
    api<TeamView>(`/systems/${name}`)
      .then(setView)
      .catch(() => setView(null))
  }, [name, finished])
  return view
}
