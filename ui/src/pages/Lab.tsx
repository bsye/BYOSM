import { Check, ChevronDown, FlaskConical, Loader2, Play, Sparkles, X } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, post, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { useLive } from "@/events"

type Method = {
  id: string
  name: string
  hypothesis: string
  code: string
  author: string
  status: "pending" | "approved" | "rejected"
  problems: string[]
  experiment: string | null
  error: string | null
  result?: { experiment: string; won: boolean | null; calib: number | null } | null
}

const STATUS = { pending: "outline", approved: "default", rejected: "secondary" } as const
const ERROR_TAIL = 400

function busy(jobs: ReturnType<typeof useLive>["jobs"], args: string[]) {
  return Object.values(jobs).some((j) => args.every((a, i) => j.args[i] === a) && (j.status === "running" || j.status === "queued"))
}

function MethodCard({ name, m, refresh }: { name: string; m: Method; refresh: () => void }) {
  const jobs = useLive().jobs
  const running = busy(jobs, ["lab", "run", name, m.id])
  const decide = (status: "approved" | "rejected") => post(`/specialists/${name}/lab/${m.id}`, { status }).then(refresh)
  const result = m.result
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {m.name}
          <Badge variant={STATUS[m.status]}>{m.status}</Badge>
          {result?.won && <Badge>sure lead</Badge>}
        </CardTitle>
        <CardDescription>{m.hypothesis}</CardDescription>
        <CardAction className="flex gap-2">
          {m.status === "pending" && (
            <>
              <Button size="sm" variant="outline" onClick={() => decide("rejected")}>
                <X />
                Reject
              </Button>
              <Button size="sm" disabled={m.problems.length > 0} onClick={() => decide("approved")}>
                <Check />
                Approve
              </Button>
            </>
          )}
          {m.status === "approved" && (
            <Button size="sm" variant="outline" disabled={running} onClick={() => runJob(["lab", "run", name, m.id])}>
              {running ? <Loader2 className="animate-spin" /> : <Play />}
              Run
            </Button>
          )}
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {m.problems.length > 0 && <p className="text-destructive">Fails its checks: {m.problems.join("; ")}</p>}
        {result && (
          <p className="text-muted-foreground">
            {result.experiment}: calib {result.calib == null ? "-" : result.calib.toFixed(4)}, {result.won ? "promoted" : "not promoted"}
          </p>
        )}
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

export function Lab({ name }: { name: string }) {
  const [methods, setMethods] = useState<Method[] | null>(null)
  const [ask, setAsk] = useState("")
  const jobs = useLive().jobs
  const writing = busy(jobs, ["lab", "write", name])
  const finished = Object.values(jobs).filter((j) => j.args[0] === "lab" && j.args[2] === name && j.status !== "running" && j.status !== "queued").length
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
            <InfoTip>
              An LLM writes a new training method as code. Nothing runs until you approve that exact code; it then trains like any experiment and is promoted
              only on a sure lead.
            </InfoTip>
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
        <MethodCard key={m.id} name={name} m={m} refresh={refresh} />
      ))}
    </div>
  )
}
