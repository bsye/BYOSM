import { Check, Save } from "lucide-react"
import { useState } from "react"
import { put } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { LlmSelect } from "@/components/llm-picker"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { compact } from "@/format"
import { OnLimitSelect } from "@/components/limit-banner"
import { roleTask } from "@/jobs"
import { matchChoice, roleConfig, useProviders, type Provider } from "@/providers"
import type { Role, Roles } from "./models-api"

const DEFAULT_ROLE = "default"
const SERVER_DEFAULT = "server-default"

function RoleRow({ r, providers, onSaved }: { r: Role; providers: Provider[] | null; onSaved: (v: Roles) => void }) {
  const [status, setStatus] = useState("")
  const save = (settings: Record<string, unknown>) =>
    put<Roles>(`/llm/roles/${r.role}`, { settings })
      .then((v) => {
        setStatus("saved")
        onSaved(v)
      })
      .catch((e) => setStatus(String(e)))
  const pickModel = (key: string) => {
    const conf = roleConfig((providers ?? []).filter((p) => p.available), key)
    if (conf) void save({ base_url: null, api_key: null, api_key_env: null, effort: null, ...conf })
  }
  return (
    <TableRow>
      <TableCell className="font-medium">
        {r.role === DEFAULT_ROLE ? "Every task" : roleTask(r.role)}
        {!r.own && r.role !== DEFAULT_ROLE && <p className="text-muted-foreground text-xs font-normal">same as every task</p>}
      </TableCell>
      <TableCell>
        <LlmSelect providers={providers} value={matchChoice(providers ?? [], r)} onChange={pickModel} label={`Model for ${r.role}`} />
      </TableCell>
      <TableCell>
        <Select value={r.effort ?? SERVER_DEFAULT} onValueChange={(v) => save({ effort: v === SERVER_DEFAULT ? null : v })}>
          <SelectTrigger size="sm" className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={SERVER_DEFAULT}>model default</SelectItem>
            {r.effort_levels.map((e) => (
              <SelectItem key={e} value={e}>
                {e}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </TableCell>
      <TableCell className="text-muted-foreground w-24 text-xs">
        {status === "saved" ? (
          <span className="flex items-center gap-1">
            <Check className="size-3.5" />
            Saved
          </span>
        ) : (
          <span className="text-destructive">{status}</span>
        )}
      </TableCell>
    </TableRow>
  )
}

export function RoleSettings({ roles, onSaved }: { roles: Roles; onSaved: (v: Roles) => void }) {
  const providers = useProviders()
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Model per task
          <InfoTip>Each task falls back to Every task. Changes apply to jobs started after them.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Task</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Effort</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {roles.roles.map((r) => (
              <RoleRow key={r.role} r={r} providers={providers} onSaved={onSaved} />
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export function LimitsCard({ roles, biggest, onSaved }: { roles: Roles; biggest: number; onSaved: (v: Roles) => void }) {
  const every = roles.roles.find((r) => r.role === DEFAULT_ROLE)
  const [budget, setBudget] = useState(String(roles.budget.tokens_per_job ?? ""))
  const [parallel, setParallel] = useState(String(every?.max_parallel ?? ""))
  const [saved, setSaved] = useState(false)
  const save = async () => {
    await put<Roles>(`/llm/roles/${DEFAULT_ROLE}`, { settings: { max_parallel: parallel ? Number(parallel) : null } })
    onSaved(await put<Roles>("/llm/budget", { tokens_per_job: budget ? Number(budget) : null }))
    setSaved(true)
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Limits
          <InfoTip>A job stops calling its LLM once it has used the token budget, input and output together. Parallel caps the calls a job sends at once; local servers take one at a time when it is empty.</InfoTip>
        </CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">largest job so far {compact(biggest)}</CardAction>
      </CardHeader>
      <CardContent className="flex flex-wrap items-end gap-4">
        <div className="space-y-2">
          <Label htmlFor="budget">Tokens per job</Label>
          <Input id="budget" className="h-8 w-40 tabular-nums" type="number" min={0} placeholder="no limit" value={budget} onChange={(e) => setBudget(e.target.value)} />
        </div>
        <OnLimitSelect />
        <div className="space-y-2">
          <Label htmlFor="parallel">Parallel calls</Label>
          <Input id="parallel" className="h-8 w-28 tabular-nums" type="number" min={1} placeholder="auto" value={parallel} onChange={(e) => setParallel(e.target.value)} />
        </div>
        <Button size="sm" className="ml-auto" onClick={save}>
          {saved ? <Check /> : <Save />}
          {saved ? "Saved" : "Save"}
        </Button>
      </CardContent>
    </Card>
  )
}
