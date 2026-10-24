import { Loader2, Play } from "lucide-react"
import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts"
import { api, runJob, type Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { StatCard, STAT_GRID } from "@/components/stat-card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useLive } from "@/events"
import { rivalName, score, signed } from "@/format"
import { openJob } from "@/job-sheet"
import { Health, type HealthReport } from "./Health"
import { RunningHere } from "./RunningHere"
import { leadOf, rivalScope, titleOf } from "./specialist-groups"

type Score = { mase: number; crps: number; calib_mase?: number; train_s?: number; ms_per_forecast?: number; published?: boolean }
type Version = { name: string; recipe: string; effort?: number; calib: number; test: number; crps: number; ci: [number, number] | null; at: number }
export type ForecasterView = Summary & {
  configs?: Record<string, Record<string, Score>>
  overall?: Record<string, Score>
  pending?: string[]
  versions?: Version[]
  health?: HealthReport | null
}

const RIVALS = ["seasonal naive", "Timestral"]
const isRival = (model: string) => RIVALS.some((r) => model.startsWith(r))

const chart = {
  mase: { label: "MASE", color: "var(--chart-1)" },
  crps: { label: "CRPS", color: "var(--chart-2)" },
} satisfies ChartConfig

export function useForecaster(name: string): ForecasterView | null | undefined {
  const [view, setView] = useState<ForecasterView | null | undefined>(undefined)
  const live = useLive()
  const progress = Object.values(live.jobs).reduce(
    (n, j) => n + (live.metrics[j.id] ?? []).filter((m) => m.kind === "series").length + (j.status === "done" ? 1 : 0),
    0,
  )
  useEffect(() => {
    api<ForecasterView>(`/series/${name}`)
      .then(setView)
      .catch(() => setView(null))
  }, [name, progress])
  return view
}

const seconds = (s?: number) => (s == null ? "-" : `${s.toFixed(1)} s`)
const millis = (s?: number) => (s == null ? "-" : `${s.toFixed(1)} ms`)

function Stats({ f }: { f: ForecasterView }) {
  const promoted = f.versions?.find((v) => v.name === f.promoted)
  if (!promoted) return null
  const recipe = f.overall?.[promoted.recipe]
  return (
    <div className={STAT_GRID}>
      <StatCard
        label={`Test ${f.metric}`}
        value={score(promoted.test)}
        detail={promoted.ci ? `95% CI ${score(promoted.ci[0])} to ${score(promoted.ci[1])}` : promoted.name}
      />
      <StatCard
        label="Lead over best rival"
        value={signed(leadOf(f))}
        detail={f.rival ? `vs ${rivalName(f.rival.name)}${rivalScope(f)}, lower is better` : "no rival scored yet"}
      />
      <StatCard label="Training" value={seconds(recipe?.train_s)} detail={`${promoted.recipe}, one model for ${Object.keys(f.configs ?? {}).length} datasets`} />
      <StatCard label="Per forecast" value={millis(recipe?.ms_per_forecast)} detail="CPU" />
    </div>
  )
}

function Board({ f, running }: { f: ForecasterView; running: boolean }) {
  const models = Object.entries(f.overall ?? {}).sort(([, a], [, b]) => a.mase - b.mase)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Against the rivals</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <ChartContainer config={chart} className="h-56 w-full">
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
              <TableHead className="text-right">Calib MASE</TableHead>
              <TableHead className="text-right">Test MASE</TableHead>
              <TableHead className="text-right">Test CRPS</TableHead>
              <TableHead className="text-right">Train</TableHead>
              <TableHead className="text-right">Per forecast</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {models.map(([name, s]) => (
              <TableRow key={name}>
                <TableCell className="font-medium">
                  {name}
                  {isRival(name) ? (
                    <Badge variant="outline" className="ml-2">
                      {s.published ? "published rival" : "rival"}
                    </Badge>
                  ) : null}
                </TableCell>
                <TableCell className="text-right tabular-nums">{score(s.calib_mase)}</TableCell>
                <TableCell className="text-right tabular-nums">{score(s.mase)}</TableCell>
                <TableCell className="text-right tabular-nums">{score(s.crps)}</TableCell>
                <TableCell className="text-right tabular-nums">{seconds(s.train_s)}</TableCell>
                <TableCell className="text-right tabular-nums">{millis(s.ms_per_forecast)}</TableCell>
              </TableRow>
            ))}
            {(running ? (f.pending ?? []) : []).map((name) => (
              <TableRow key={name}>
                <TableCell className="text-muted-foreground flex items-center gap-2">
                  <Loader2 className="size-3.5 animate-spin" />
                  {name}
                </TableCell>
                <TableCell colSpan={5} className="text-muted-foreground text-right">
                  training
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Versions({ f }: { f: ForecasterView }) {
  return (
    <Card>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Version</TableHead>
              <TableHead>Recipe</TableHead>
              <TableHead className="text-right">Training</TableHead>
              <TableHead className="text-right">Calib MASE</TableHead>
              <TableHead className="text-right">Test MASE</TableHead>
              <TableHead className="text-right">95% CI</TableHead>
              <TableHead className="text-right">Built</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[...(f.versions ?? [])].reverse().map((v) => (
              <TableRow key={v.name}>
                <TableCell className="font-medium">
                  {v.name}
                  {v.name === f.promoted && <Badge className="ml-2">promoted</Badge>}
                </TableCell>
                <TableCell>{v.recipe}</TableCell>
                <TableCell className="text-right tabular-nums">{2 ** (v.effort ?? 0)}x</TableCell>
                <TableCell className="text-right tabular-nums">{score(v.calib)}</TableCell>
                <TableCell className="text-right tabular-nums">{score(v.test)}</TableCell>
                <TableCell className="text-right tabular-nums">{v.ci ? `${score(v.ci[0])} to ${score(v.ci[1])}` : "-"}</TableCell>
                <TableCell className="text-right text-xs tabular-nums">
                  {new Date(v.at * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Datasets({ f }: { f: ForecasterView }) {
  const models = Object.keys(f.overall ?? {})
  return (
    <Card>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Dataset</TableHead>
              {models.map((m) => (
                <TableHead key={m} className="text-right">
                  {m}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(f.configs ?? {}).map(([config, row]) => (
              <TableRow key={config}>
                <TableCell className="font-medium">{config}</TableCell>
                {models.map((m) => (
                  <TableCell key={m} className="text-right tabular-nums">
                    {score(row[m]?.mase)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export function Forecaster({ f }: { f: ForecasterView }) {
  const running = Object.values(useLive().jobs).some((j) => j.args[0] === "series" && j.args[2] === f.name && (j.status === "running" || j.status === "queued"))
  const train = () => void runJob(["series", "run", f.name, "--until-rival"]).then((j) => openJob(j.id))
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            {titleOf(f)}
            <InfoTip>
              One model per recipe, trained on all of these GIFT-Eval datasets together. MASE and CRPS are relative to seasonal naive (1.0, the dashed line):
              lower is better. Each run picks the recipe on the calib windows and reports the test windows. Published scores come from the GIFT-Eval leaderboard
              and show only where our seasonal naive matches theirs.
            </InfoTip>
          </h1>
          <p className="text-muted-foreground text-sm">{f.description}</p>
        </div>
        <Button onClick={train}>
          <Play />
          Keep training
        </Button>
      </div>
      <RunningHere name={f.name} />
      <Health h={f.health} />
      <Stats f={f} />
      {f.overall && Object.keys(f.overall).length ? (
        <Tabs defaultValue="board">
          <TabsList>
            <TabsTrigger value="board">Board</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
            <TabsTrigger value="datasets">Datasets</TabsTrigger>
          </TabsList>
          <TabsContent value="board">
            <Board f={f} running={running} />
          </TabsContent>
          <TabsContent value="versions">
            <Versions f={f} />
          </TabsContent>
          <TabsContent value="datasets">
            <Datasets f={f} />
          </TabsContent>
        </Tabs>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>Not run yet</EmptyTitle>
            <EmptyDescription>Keep training builds the first version and scores it against the rivals.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      )}
    </div>
  )
}
