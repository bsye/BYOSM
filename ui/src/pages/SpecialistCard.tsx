import { Loader2, Minus, Play, TrendingDown, TrendingUp } from "lucide-react"
import { runJob, type Summary } from "@/api"
import { Button } from "@/components/ui/button"
import { openJob } from "@/job-sheet"
import type { NextStep } from "./NextSteps"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { rivalName, score, signed } from "@/format"
import { href } from "@/route"
import { createdLabel, kindOf, leadOf, rivalScope, titleOf } from "./specialist-groups"

const TIE = 0.0005

function Lead({ s }: { s: Summary }) {
  const d = leadOf(s)
  if (d == null || !s.rival) return <span className="text-muted-foreground">No rival scored yet</span>
  const tied = Math.abs(d) < TIE
  const Icon = tied ? Minus : d > 0 ? TrendingUp : TrendingDown
  return (
    <span className="text-muted-foreground flex min-w-0 items-center gap-1">
      <Icon className="size-3.5 shrink-0" />
      <span className="text-foreground font-medium tabular-nums">{tied ? "tied" : signed(d)}</span>
      <span className="min-w-0 truncate">vs {rivalName(s.rival.name)}{rivalScope(s)}</span>
    </span>
  )
}

function Status({ s, running }: { s: Summary; running?: string }) {
  if (running)
    return (
      <span className="flex items-center gap-1.5">
        <Loader2 className="size-3.5 animate-spin" />
        {running}
      </span>
    )
  if (s.kind === "team") return <span>{s.members?.length === 1 ? "1 model" : `${s.members?.length} models`}</span>
  if (s.kind === "risk") return <span>{s.score == null ? "Not run yet" : s.exported ? "Scored today" : "Tested"}</span>
  if (s.kind === "forecast") return <span>{s.promoted ? `${s.promoted}, ${s.experiments.length} versions` : "Not run yet"}</span>
  return <span>{s.exported ? "Exported" : s.score == null ? "Not built yet" : "Not exported"}</span>
}

function Step({ steps }: { steps: NextStep[] }) {
  const [first] = steps
  const run = (e: React.MouseEvent) => {
    e.stopPropagation()
    void runJob(first.args).then((j) => openJob(j.id))
  }
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="text-foreground min-w-0 truncate" title={first.why}>
        {first.title}
      </span>
      <Button size="xs" variant="outline" onClick={run}>
        <Play />
        Run
      </Button>
    </div>
  )
}

export function SpecialistCard({ s, running, showKind, steps }: { s: Summary; running?: string; showKind: boolean; steps?: NextStep[] }) {
  const kind = kindOf(s)
  const open = () => {
    window.location.hash = href("specialists", s.name)
  }
  return (
    <div role="link" tabIndex={0} onClick={open} onKeyDown={(e) => e.key === "Enter" && open()} className="h-full cursor-pointer">
      <Card className="hover:bg-accent/30 h-full gap-4 transition-colors">
        <CardHeader>
          <CardTitle className="truncate">{titleOf(s)}</CardTitle>
          <CardDescription className="truncate font-mono text-xs">{s.name}</CardDescription>
          {showKind && (
            <CardAction>
              <Badge variant="outline">
                <kind.icon />
                {kind.label}
              </Badge>
            </CardAction>
          )}
        </CardHeader>
        <CardContent className="mt-auto space-y-1">
          {s.score == null ? (
            <p className="text-muted-foreground text-3xl font-semibold">-</p>
          ) : (
            <p className="flex items-baseline gap-2">
              <span className="text-3xl font-semibold tabular-nums">{score(s.score)}</span>
              <span className="text-muted-foreground text-xs">test {s.metric}</span>
            </p>
          )}
          <div className="text-xs">{s.score != null && <Lead s={s} />}</div>
        </CardContent>
        <CardFooter className="text-muted-foreground/80 justify-between gap-3 text-xs">
          <span className="flex shrink-0 items-center gap-1.5">
            <Status s={s} running={running} />
            {s.created && (
              <>
                <span>·</span>
                <span className="tabular-nums">{createdLabel(s)}</span>
              </>
            )}
          </span>
          {steps && steps.length > 0 && <Step steps={steps} />}
        </CardFooter>
      </Card>
    </div>
  )
}
