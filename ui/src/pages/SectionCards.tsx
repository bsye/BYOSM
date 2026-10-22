import { Loader2, Play } from "lucide-react"
import { useState } from "react"
import type { Summary } from "@/api"
import { LlmSelect, useStandardChoice } from "@/components/llm-picker"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import type { Health, Healths } from "./HealthList"
import { bestSteps, unblock, untilReady } from "./stat-actions"

const PRIMARY = 0

type Llm = Record<string, unknown> | null
type Stat = { label: string; value: string; badge?: string; line: string; hint?: string; action?: { label: string; run: (llm: Llm) => Promise<void> } }

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
  const notReady = known.filter((k) => !k.h.ready).map((k) => k.s.name)
  const blocked = known.filter((k) => blocker && k.h.blockers.includes(blocker)).map((k) => k.s.name)
  return [
    {
      label: "Ready for production",
      value: `${ready} / ${items.length}`,
      badge: items.length ? `${Math.round((ready / items.length) * 100)}%` : undefined,
      line: pending ? `checking ${pending} more` : `${items.length - ready} need work`,
      action: notReady.length ? { label: "Auto", run: (llm) => untilReady(notReady, llm) } : undefined,
    },
    {
      label: "Average health",
      value: mean == null ? "-" : String(mean),
      badge: mean == null ? undefined : `ready: ${known[0].h.ready_score}+ and unblocked`,
      line: `across ${known.length} models`,
      action: known.length ? { label: "Run all", run: (llm) => bestSteps(known.map((k) => k.s.name), llm) } : undefined,
    },
    {
      label: "Closest to ready",
      value: close ? String(close.h.score) : "-",
      line: close ? close.s.title || close.s.name : "every model is ready",
      hint: close ? `blocked by ${close.h.blockers.join(", ") || "a low score"}` : undefined,
      action: close ? { label: "Auto", run: (llm) => untilReady([close.s.name], llm) } : undefined,
    },
    {
      label: "Top blocker",
      value: blocker ?? "None",
      line: blocker ? `holds back ${blocks} of ${known.length} models` : "nothing blocks production",
      action: blocker ? { label: "Fix all", run: (llm) => unblock(blocked, blocker, llm) } : undefined,
    },
  ]
}

function Action({ action }: { action: NonNullable<Stat["action"]> }) {
  const llm = useStandardChoice()
  const [busy, setBusy] = useState(false)
  const start = () => {
    setBusy(true)
    action.run(llm.config()).finally(() => setBusy(false))
  }
  return (
    <div className="flex w-full items-center gap-2">
      <LlmSelect providers={llm.providers} value={llm.value} onChange={llm.pick} label="LLM for this action" standard className="min-w-0 flex-1" />
      <Button size="sm" disabled={busy} onClick={start}>
        {busy ? <Loader2 className="animate-spin" /> : <Play />}
        {action.label}
      </Button>
    </div>
  )
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
          <CardFooter className="text-muted-foreground mt-auto flex-col items-start gap-3 text-sm">
            <span className="line-clamp-1" title={s.hint}>
              {s.line}
            </span>
            {s.action && <Action action={s.action} />}
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
