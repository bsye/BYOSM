import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { gigabytes, seconds, compact } from "@/format"
import { modelName, roleTask } from "@/jobs"
import { byTaskAndModel, useModels, type Usage } from "./models-api"
import { LocalModels, ModelsInUse } from "./ModelsRunning"
import { LimitsCard, RoleSettings } from "./ModelsRoles"

const counted = (u: Usage, n: number) => (u.tokens_in + u.tokens_out ? compact(n) : "-")

const sum = (rows: Usage[], f: (u: Usage) => number) => rows.reduce((n, u) => n + f(u), 0)

function UsageChart({ rows }: { rows: Usage[] }) {
  const tasks = new Map<string, { task: string; total: number; calls: number }>()
  for (const u of rows) {
    const t = tasks.get(u.role) ?? { task: roleTask(u.role), total: 0, calls: 0 }
    tasks.set(u.role, { ...t, total: t.total + u.tokens_in + u.tokens_out, calls: t.calls + u.calls })
  }
  const ranked = [...tasks.values()].filter((t) => t.total > 0).sort((a, b) => b.total - a.total)
  const all = sum(rows, (u) => u.tokens_in + u.tokens_out)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Tokens by task</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3">
        {ranked.map((t) => (
          <div key={t.task} className="grid grid-cols-[10rem_1fr_7rem] items-center gap-3 text-sm">
            <span className="truncate font-medium">{t.task}</span>
            <Progress value={(t.total / all) * 100} />
            <span className="text-muted-foreground text-right tabular-nums">
              {compact(t.total)} <span className="text-muted-foreground/60">{Math.round((t.total / all) * 100)}%</span>
            </span>
          </div>
        ))}
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
              <TableHead className="text-right">Tokens per call</TableHead>
              <TableHead className="text-right">Output tokens/s</TableHead>
              <TableHead className="text-right">Time per call</TableHead>
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
                <TableCell className="text-right tabular-nums">{u.counted ? compact(Math.round((u.tokens_in + u.tokens_out) / u.counted)) : "-"}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {u.counted && u.counted_seconds ? (u.tokens_out / u.counted_seconds).toFixed(1) : "-"}
                </TableCell>
                <TableCell className="text-foreground/70 text-right tabular-nums">{seconds(u.seconds / Math.max(1, u.calls))}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export function Trainers() {
  const m = useModels()
  const total = sum(m.usage, (u) => u.tokens_in + u.tokens_out)
  const perJob = new Map<string, number>()
  for (const u of m.usage) perJob.set(u.job, (perJob.get(u.job) ?? 0) + u.tokens_in + u.tokens_out)
  const biggest = Math.max(0, ...perJob.values())
  const budget = m.roles?.budget.tokens_per_job
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Trainers</h1>
      <div className={STAT_GRID}>
        <StatCard label="Tokens" value={compact(total)} detail={`${compact(sum(m.usage, (u) => u.tokens_out))} output`} />
        <StatCard
          label="LLM calls"
          value={String(sum(m.usage, (u) => u.calls))}
          detail={`${perJob.size} jobs, ${compact(
            Math.round(
              total /
                Math.max(
                  1,
                  sum(m.usage, (u) => u.counted),
                ),
            ),
          )} tokens per call`}
        />
        <StatCard label="Loaded locally" value={String(m.loaded?.length ?? "-")} detail={gigabytes(m.loaded?.reduce((n, x) => n + (x.bytes ?? 0), 0))} />
        <StatCard label="Budget per job" value={budget ? compact(budget) : "None"} detail={`largest job ${compact(biggest)}`} />
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
