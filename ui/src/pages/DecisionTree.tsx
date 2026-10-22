import { Bot, CircleCheck, Cog, GitBranch, Play } from "lucide-react"
import { useState } from "react"
import { runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { question, type Alternative, type Decision } from "@/job-story"
import { modelName } from "@/jobs"
import { openJob } from "@/job-sheet"

const RULE = "rule"

function Who({ by }: { by?: string }) {
  if (!by) return null
  const Icon = by === RULE ? Cog : Bot
  return (
    <Badge variant="outline" className="text-muted-foreground font-normal">
      <Icon />
      {by === RULE ? "rule" : modelName(by)}
    </Badge>
  )
}

function Branch({ alt }: { alt: Alternative }) {
  const [error, setError] = useState("")
  const run = () =>
    alt.args &&
    runJob(alt.args)
      .then((j) => openJob(j.id))
      .catch((e) => setError(String(e)))
  return (
    <li className="flex items-start gap-2">
      <GitBranch className="text-muted-foreground mt-0.5 size-3.5 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-foreground/80 truncate text-sm">{alt.option}</p>
        <p className="text-muted-foreground text-xs">{error || alt.why_not}</p>
      </div>
      {alt.args && (
        <Button size="xs" variant="outline" onClick={run}>
          <Play />
          Try
        </Button>
      )}
    </li>
  )
}

function Node({ d }: { d: Decision }) {
  const choice = d.choice == null || d.choice === "" ? "None" : String(d.choice)
  return (
    <li className="relative border-l pb-5 pl-5 last:pb-0">
      <span className="bg-foreground ring-background absolute top-1 -left-[5px] size-2.5 rounded-full ring-4" />
      <p className="text-muted-foreground text-xs">{question(d.step)}</p>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <CircleCheck className="size-4 shrink-0" />
        <span className="font-medium break-all">{choice}</span>
        <Who by={d.by} />
      </div>
      {d.why && <p className="text-muted-foreground mt-1 text-xs">{d.why}</p>}
      {d.alternatives.length > 0 && (
        <ul className="mt-3 space-y-2 border-l border-dashed pl-4">
          {d.alternatives.map((a) => (
            <Branch key={a.option} alt={a} />
          ))}
        </ul>
      )}
    </li>
  )
}

export function DecisionTree({ items }: { items: Decision[] }) {
  const paths = items.reduce((n, d) => n + d.alternatives.length, 0)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Decisions
          <InfoTip>Every choice the autopilot made, who made it, and the paths it did not take. Try runs one as a new experiment.</InfoTip>
        </CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">
          {items.length} made{paths ? `, ${paths} not taken` : ""}
        </CardAction>
      </CardHeader>
      <CardContent>
        <ol className="ml-1">
          {items.map((d, i) => (
            <Node key={i} d={d} />
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}
