import type { Summary } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import type { Health, Healths } from "./HealthList"

const PRIMARY = 0

type Stat = { label: string; value: string; badge?: string; line: string }

function counted<T>(xs: T[]): [T, number][] {
  const c = new Map<T, number>()
  for (const x of xs) c.set(x, (c.get(x) ?? 0) + 1)
  return [...c.entries()].sort((a, b) => b[1] - a[1])
}

function stats(items: Summary[], healths: Healths): Stat[] {
  const known = items.flatMap((s) => (healths[s.name] ? [{ s, h: healths[s.name] as Health }] : []))
  const pending = items.filter((s) => healths[s.name] === undefined).length
  const ready = known.filter((k) => k.h.ready).length
  const mean = known.length ? Math.round(known.reduce((n, k) => n + k.h.score, 0) / known.length) : null
  const close = known.filter((k) => !k.h.ready).sort((a, b) => b.h.score - a.h.score)[0]
  const [blocker, blocks] = counted(known.flatMap((k) => k.h.blockers))[0] ?? [null, 0]
  return [
    {
      label: "Ready for production",
      value: `${ready} / ${items.length}`,
      badge: items.length ? `${Math.round((ready / items.length) * 100)}%` : undefined,
      line: pending ? `checking ${pending} more` : `${items.length - ready} need work`,
    },
    {
      label: "Average health",
      value: mean == null ? "-" : String(mean),
      badge: mean == null ? undefined : `ready at ${known[0].h.ready_score}`,
      line: `across ${known.length} models`,
    },
    {
      label: "Closest to ready",
      value: close ? String(close.h.score) : "-",
      line: close ? `${close.s.title || close.s.name}, blocked by ${close.h.blockers[0] ?? "a low score"}` : "every model is ready",
    },
    {
      label: "Top blocker",
      value: blocker ?? "None",
      line: blocker ? `holds back ${blocks} of ${known.length} models` : "nothing blocks production",
    },
  ]
}

export function SectionCards({ items, healths }: { items: Summary[]; healths: Healths }) {
  return (
    <div className="grid auto-rows-fr grid-cols-2 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs @5xl/main:grid-cols-4 dark:*:data-[slot=card]:bg-card">
      {stats(items, healths).map((s, i) => (
        <Card key={s.label} className="@container/card h-full">
          <CardHeader>
            <CardDescription className="line-clamp-1">{s.label}</CardDescription>
            <CardTitle className={i === PRIMARY ? "text-3xl font-semibold tabular-nums @[250px]/card:text-4xl" : "line-clamp-1 text-2xl font-semibold tabular-nums"}>
              {s.value}
            </CardTitle>
            {s.badge && (
              <CardAction>
                <Badge variant="outline">{s.badge}</Badge>
              </CardAction>
            )}
          </CardHeader>
          <CardFooter className="text-muted-foreground mt-auto text-sm">
            <span className="line-clamp-1">{s.line}</span>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
