import { TrendingDown, TrendingUp } from "lucide-react"
import type { Summary } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { useLive } from "@/events"
import { rivalName } from "@/format"

const PRIMARY = 0

type Stat = { label: string; value: string; badge?: { up?: boolean; text: string }; line: string }

function stats(items: Summary[], running: number): Stat[] {
  const scored = items.filter((s) => s.score != null && s.rival?.score != null)
  const margins = scored.map((s) => ({ name: s.name, rival: s.rival?.name, d: (s.score ?? 0) - (s.rival?.score ?? 0) }))
  const ahead = margins.filter((m) => m.d > 0).length
  const best = [...margins].sort((a, b) => b.d - a.d)[0]
  return [
    {
      label: "Ahead of the best rival",
      value: `${ahead} / ${scored.length}`,
      badge: { text: `${Math.round((ahead / Math.max(1, scored.length)) * 100)}%` },
      line: "public benchmarks, test",
    },
    {
      label: "Biggest lead",
      value: best ? `+${best.d.toFixed(3)}` : "-",
      line: best ? `${best.name} vs ${rivalName(best.rival)}` : "nothing scored yet",
    },
    {
      label: "Specialists",
      value: String(items.length),
      line: `${items.filter((s) => s.exported).length} exported`,
    },
    {
      label: "Running jobs",
      value: String(running),
      line: running ? "streaming live" : "idle",
    },
  ]
}

export function SectionCards({ items }: { items: Summary[] }) {
  const running = Object.values(useLive().jobs).filter((j) => j.status === "running" || j.status === "queued").length
  return (
    <div className="grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs @xl/main:grid-cols-2 @5xl/main:grid-cols-4 dark:*:data-[slot=card]:bg-card">
      {stats(items, running).map((s, i) => (
        <Card key={s.label} className="@container/card h-full">
          <CardHeader>
            <CardDescription className="line-clamp-1">{s.label}</CardDescription>
            <CardTitle className={i === PRIMARY ? "text-3xl font-semibold tabular-nums @[250px]/card:text-4xl" : "text-2xl font-semibold tabular-nums"}>
              {s.value}
            </CardTitle>
            {s.badge && (
              <CardAction>
                <Badge variant="outline">
                  {s.badge.up === true && <TrendingUp />}
                  {s.badge.up === false && <TrendingDown />}
                  {s.badge.text}
                </Badge>
              </CardAction>
            )}
          </CardHeader>
          <CardFooter className="mt-auto flex-col items-start gap-1 text-sm">
            <div className="text-muted-foreground line-clamp-1">{s.line}</div>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
