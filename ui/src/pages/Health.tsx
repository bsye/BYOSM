import { CircleCheck, CircleDashed, CircleX, HeartPulse, TriangleAlert, type LucideIcon } from "lucide-react"
import type { Detail } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export type Status = "ok" | "warn" | "fail" | "unknown"
export type Check = { id: string; title: string; weight: number; status: Status; value: string; why: string }
export type HealthReport = { score: number; ready: boolean; ready_score: number; blockers: string[]; checks: Check[] }

const ICON: Record<Status, { icon: LucideIcon; tone: string; label: string }> = {
  ok: { icon: CircleCheck, tone: "text-foreground/70", label: "Pass" },
  warn: { icon: TriangleAlert, tone: "text-amber-600 dark:text-amber-500", label: "Warning" },
  fail: { icon: CircleX, tone: "text-destructive", label: "Fail" },
  unknown: { icon: CircleDashed, tone: "text-muted-foreground", label: "Not measured" },
}

function Row({ c }: { c: Check }) {
  const s = ICON[c.status]
  return (
    <li className="flex items-center gap-2 py-2">
      <s.icon className={`size-4 shrink-0 ${s.tone}`} aria-label={s.label} />
      <span className="font-medium">{c.title}</span>
      <InfoTip>{c.why}</InfoTip>
      <span className={`ml-auto truncate text-right text-sm tabular-nums ${c.status === "ok" ? "text-muted-foreground" : s.tone}`}>{c.value}</span>
    </li>
  )
}

export function Health({ d }: { d: Detail }) {
  const h = d.health as HealthReport | null
  if (!h) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HeartPulse className="size-4" />
          Health
        </CardTitle>
        <CardAction>
          <Badge variant={h.ready ? "default" : "outline"}>{h.ready ? "Production ready" : "Not ready"}</Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-baseline gap-1">
            <span className="text-4xl font-semibold tabular-nums">{h.score}</span>
            <span className="text-muted-foreground text-sm">/ 100</span>
            <span className="text-muted-foreground ml-auto truncate text-xs">
              {h.ready ? `${h.ready_score}+ and no failing check` : h.blockers.length ? `Blocked: ${h.blockers.join(", ")}` : `Needs ${h.ready_score}+`}
            </span>
          </div>
          <Progress value={h.score} aria-label="Health score" />
        </div>
        <ul className="divide-y border-t">
          {h.checks.map((c) => (
            <Row key={c.id} c={c} />
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}
