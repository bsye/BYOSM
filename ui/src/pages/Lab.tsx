import { ChevronDown, FlaskConical, Loader2, Sparkles } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { useFinished, useLive } from "@/events"
import { jobRunning } from "@/jobs"
import { MethodActions, type MethodStatus } from "./MethodActions"

type Method = {
  id: string
  name: string
  hypothesis: string
  code: string
  author: string
  status: MethodStatus
  problems: string[]
  experiment: string | null
  error: string | null
  refused?: string | null
  result?: { experiment: string; won: boolean | null; calib: number | null } | null
  parent?: string | null
  paper?: string | null
}

type LabKind = "classifier" | "forecaster"

const STATUS = { pending: "outline", approved: "default", rejected: "secondary" } as const
const PROMOTION: Record<LabKind, { rule: string; badge: string; score: string }> = {
  classifier: { rule: "it then trains like any experiment and is promoted only on a sure lead", badge: "sure lead", score: "calib" },
  forecaster: { rule: "it then refits the patch transformers, and the new version is promoted only on a lower calib MASE", badge: "", score: "calib MASE" },
}
const ERROR_TAIL = 400

function MethodCard({ name, kind, m, refresh }: { name: string; kind: LabKind; m: Method; refresh: () => void }) {
  const result = m.result
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {m.name}
          <Badge variant={STATUS[m.status]}>{m.status}</Badge>
          {result?.won && PROMOTION[kind].badge && <Badge>{PROMOTION[kind].badge}</Badge>}
          {m.parent && <Badge variant="outline">revision of {m.parent}</Badge>}
        </CardTitle>
        <CardDescription>
          {m.hypothesis}
          {m.paper && (
            <a href={m.paper} target="_blank" rel="noreferrer" className="hover:text-foreground ml-1 underline">
              paper
            </a>
          )}
        </CardDescription>
        <CardAction>
          <MethodActions owner={name} id={m.id} status={m.status} problems={m.problems} onDecided={refresh} />
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {m.problems.length > 0 && <p className="text-destructive">Fails its checks: {m.problems.join("; ")}</p>}
        {result && (
          <p className="text-muted-foreground">
            {result.experiment}: {PROMOTION[kind].score} {result.calib == null ? "-" : result.calib.toFixed(4)}, {result.won ? "promoted" : "not promoted"}
          </p>
        )}
        {m.refused && <p className="text-muted-foreground">Refused: {m.refused}</p>}
        {m.error && <p className="text-destructive font-mono text-xs whitespace-pre-wrap">{m.error.slice(-ERROR_TAIL)}</p>}
        <Collapsible>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="-ml-2">
              <ChevronDown />
              Code by {m.author}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="bg-muted overflow-x-auto rounded-md p-3 font-mono text-xs">{m.code}</pre>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  )
}

export function Lab({ name, kind }: { name: string; kind: LabKind }) {
  const [methods, setMethods] = useState<Method[] | null>(null)
  const [ask, setAsk] = useState("")
  const jobs = useLive().jobs
  const writing = jobRunning(jobs, ["lab", "write", name])
  const finished = useFinished()
  const refresh = useCallback(() => api<Method[]>(`/specialists/${name}/lab`).then(setMethods, () => setMethods([])), [name])
  useEffect(() => {
    refresh()
  }, [refresh, finished])
  const write = () => runJob(ask.trim() ? ["lab", "write", name, "--ask", ask.trim()] : ["lab", "write", name]).then(() => setAsk(""))
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Lab
            <InfoTip>An LLM writes a new training method as code. Nothing runs until you approve that exact code; {PROMOTION[kind].rule}.</InfoTip>
          </CardTitle>
          <CardAction>
            <Button onClick={write} disabled={writing}>
              {writing ? <Loader2 className="animate-spin" /> : <Sparkles />}
              Write a method
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>
          <Input value={ask} placeholder="Optional: what the method should try" onChange={(e) => setAsk(e.target.value)} />
        </CardContent>
      </Card>
      {methods && methods.length === 0 && !writing && (
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <FlaskConical />
            </EmptyMedia>
            <EmptyTitle>No methods yet</EmptyTitle>
            <EmptyDescription>Write one to see its code here before it runs.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      )}
      {methods?.map((m) => (
        <MethodCard key={m.id} name={name} kind={kind} m={m} refresh={refresh} />
      ))}
    </div>
  )
}
