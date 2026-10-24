import { CircleCheck, CircleDashed, CircleX, Loader2 } from "lucide-react"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { seconds } from "@/format"
import type { Stage, StageState } from "@/job-story"
import type { Step } from "@/jobs"

const ICONS = { running: Loader2, done: CircleCheck, failed: CircleX, waiting: CircleDashed }

function Icon({ state }: { state: StageState | "waiting" }) {
  const I = ICONS[state]
  const tone = state === "failed" ? "text-destructive" : state === "running" ? "" : "text-muted-foreground"
  return <I className={`size-4 shrink-0 ${tone} ${state === "running" ? "animate-spin" : ""}`} />
}

function SubStep({ step, now }: { step: Step; now: number }) {
  const live = step.state === "running" && step.at ? ` · ${seconds(now - step.at)}` : ""
  return (
    <li className="flex items-center gap-2 text-xs">
      <Icon state={step.state} />
      <span className="text-foreground/80 truncate">{step.label}</span>
      <span className="text-muted-foreground ml-auto shrink-0 tabular-nums">
        {step.detail}
        {live}
      </span>
    </li>
  )
}

function Row({ stage, now }: { stage: Stage; now: number }) {
  const time = stage.seconds != null ? seconds(stage.seconds) : seconds(now - stage.start)
  return (
    <li className="py-2.5">
      <div className="flex items-center gap-3">
        <Icon state={stage.state} />
        <span className={stage.state === "running" ? "font-medium" : "text-foreground/80"}>{stage.name}</span>
        <span className="text-muted-foreground ml-auto text-xs tabular-nums">{time}</span>
      </div>
      {stage.steps.length > 0 && (
        <ul className="mt-2 ml-7 space-y-1.5">
          {stage.steps.map((s, i) => (
            <SubStep key={i} step={s} now={now} />
          ))}
        </ul>
      )}
    </li>
  )
}

export function JobStages({ items, started, active, now }: { items: Stage[]; started: number | null; active: boolean; now: number }) {
  const booting = active && !items.length
  const total = started ? seconds(now - started) : ""
  return (
    <Card>
      <CardHeader>
        <CardTitle>Stages</CardTitle>
        <CardAction className="text-muted-foreground text-sm tabular-nums">{total}</CardAction>
      </CardHeader>
      <CardContent>
        <ol className="divide-y">
          {booting && (
            <Row
              stage={{ name: started ? "Starting: loading the model libraries" : "Queued", state: "running", start: started ?? now, seconds: null, steps: [] }}
              now={now}
            />
          )}
          {items.map((s, i) => (
            <Row key={i} stage={s} now={now} />
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}
