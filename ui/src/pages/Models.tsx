import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { gigabytes, seconds, tokens } from "@/format"
import { modelName, roleTask } from "@/jobs"
import { byTaskAndModel, useModels, type Usage } from "./models-api"
import { LocalModels, ModelsInUse } from "./ModelsRunning"
import { LimitsCard, RoleSettings } from "./ModelsRoles"

const config = {
  tokens_in: { label: "input", color: "var(--chart-1)" },
  tokens_out: { label: "output", color: "var(--chart-3)" },
} satisfies ChartConfig

const counted = (u: Usage, n: number) => (u.tokens_in + u.tokens_out ? tokens(n) : "-")

const sum = (rows: Usage[], f: (u: Usage) => number) => rows.reduce((n, u) => n + f(u), 0)

function UsageChart({ rows }: { rows: Usage[] }) {
  const tasks = new Map<string, { task: string; tokens_in: number; tokens_out: number }>()
  for (const u of rows) {
    const t = tasks.get(u.role) ?? { task: roleTask(u.role), tokens_in: 0, tokens_out: 0 }
    tasks.set(u.role, { ...t, tokens_in: t.tokens_in + u.tokens_in, tokens_out: t.tokens_out + u.tokens_out })
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Tokens by task</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="h-56 w-full">
          <BarChart data={[...tasks.values()]}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="task" tickLine={false} axisLine={false} />
            <YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v) => tokens(Number(v))} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Bar dataKey="tokens_in" stackId="t" fill="var(--color-tokens_in)" />
            <Bar dataKey="tokens_out" stackId="t" fill="var(--color-tokens_out)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

function UsageTable({ rows }: { rows: Usage[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Per task and model</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Task</TableHead>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Jobs</TableHead>
              <TableHead className="text-right">Calls</TableHead>
              <TableHead className="text-right">Input</TableHead>
              <TableHead className="text-right">Output</TableHead>
              <TableHead className="text-right">LLM time</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {byTaskAndModel(rows).map((u) => (
              <TableRow key={`${u.role}-${u.model}`}>
                <TableCell className="font-medium">{roleTask(u.role)}</TableCell>
                <TableCell className="text-foreground/70">{modelName(u.model)}</TableCell>
                <TableCell className="text-right tabular-nums">{u.jobs}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {u.calls}
                  {u.failed > 0 && <span className="text-destructive ml-1 text-xs">+{u.failed} failed</span>}
                </TableCell>
                <TableCell className="text-right tabular-nums">{counted(u, u.tokens_in)}</TableCell>
                <TableCell className="text-right tabular-nums">{counted(u, u.tokens_out)}</TableCell>
                <TableCell className="text-foreground/70 text-right tabular-nums">{seconds(u.seconds)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export function Models() {
  const m = useModels()
  const total = sum(m.usage, (u) => u.tokens_in + u.tokens_out)
  const perJob = new Map<string, number>()
  for (const u of m.usage) perJob.set(u.job, (perJob.get(u.job) ?? 0) + u.tokens_in + u.tokens_out)
  const biggest = Math.max(0, ...perJob.values())
  const budget = m.roles?.budget.tokens_per_job
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Models</h1>
      <div className={STAT_GRID}>
        <StatCard label="Tokens" value={tokens(total)} detail={`${tokens(sum(m.usage, (u) => u.tokens_out))} output`} />
        <StatCard label="LLM calls" value={String(sum(m.usage, (u) => u.calls))} detail={`${perJob.size} jobs`} />
        <StatCard label="Loaded locally" value={String(m.loaded?.length ?? "-")} detail={gigabytes(m.loaded?.reduce((n, x) => n + (x.bytes ?? 0), 0))} />
        <StatCard label="Budget per job" value={budget ? tokens(budget) : "None"} detail={`largest job ${tokens(biggest)}`} />
      </div>
      <div className="grid gap-4 md:gap-6 @5xl/main:grid-cols-2">
        <LocalModels items={m.loaded} onChange={m.setLoaded} />
        <ModelsInUse />
      </div>
      {total > 0 && <UsageChart rows={m.usage} />}
      {m.usage.length > 0 && <UsageTable rows={m.usage} />}
      {m.roles && <RoleSettings roles={m.roles} onSaved={m.setRoles} />}
      {m.roles && <LimitsCard roles={m.roles} biggest={biggest} onSaved={m.setRoles} />}
    </div>
  )
}
