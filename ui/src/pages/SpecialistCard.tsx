import { Loader2, Minus, TrendingDown, TrendingUp } from "lucide-react"
import type { Summary } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { metricName, rivalName, score, signed } from "@/format"
import { href } from "@/route"
import { createdLabel, kindOf, titleOf } from "./specialist-groups"

function Lead({ s }: { s: Summary }) {
  if (s.score == null || s.rival?.score == null) return <span className="text-muted-foreground">No rival scored yet</span>
  const d = s.score - s.rival.score
  const Icon = d > 0 ? TrendingUp : d < 0 ? TrendingDown : Minus
  return (
    <span className="text-muted-foreground flex min-w-0 items-center gap-1">
      <Icon className="size-3.5 shrink-0" />
      <span className="text-foreground font-medium tabular-nums">{signed(d)}</span>
      <span className="min-w-0 truncate">vs {rivalName(s.rival.name)}</span>
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
  return <span>{s.exported ? "Exported" : s.score == null ? "Not built yet" : "Not exported"}</span>
}

export function SpecialistCard({ s, running, showKind }: { s: Summary; running?: string; showKind: boolean }) {
  const kind = kindOf(s)
  return (
    <a href={href("specialists", s.name)} className="h-full">
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
              <span className="text-muted-foreground text-xs">test {metricName(s.metric)}</span>
            </p>
          )}
          <div className="text-xs">{s.score != null && <Lead s={s} />}</div>
        </CardContent>
        <CardFooter className="text-muted-foreground/80 justify-between text-xs">
          <Status s={s} running={running} />
          <span className="tabular-nums">{createdLabel(s)}</span>
        </CardFooter>
      </Card>
    </a>
  )
}
