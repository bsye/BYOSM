import { Loader2 } from "lucide-react"
import { useState } from "react"
import { post } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Switch } from "@/components/ui/switch"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { featureValue } from "./forecast-api"

type Effect = { feature: string; drop: number; value: string | number | null; typical: string | number | null }
export type Risk = { state: "alive" | "ended" | "unscored"; row: Record<string, string | null>; chance?: number; usual?: string[]; effects?: Effect[] }

const MIN_SCALE = 0.01
const pct = (v: number) => `${Math.round(v * 100)}%`
const signedPoints = (v: number) => (Math.round(v * 1000) / 10 || 0).toLocaleString(undefined, { signDisplay: "exceptZero", minimumFractionDigits: 1 })
const points = (v: number) => `${signedPoints(v)} pts`

function EffectBar({ drop, widest }: { drop: number; widest: number }) {
  const half = (Math.abs(drop) / widest) * 50
  return (
    <div className="bg-muted relative h-2 w-full rounded-full">
      <div className="bg-border absolute inset-y-0 left-1/2 w-px" />
      <div className="bg-primary/70 absolute inset-y-0 rounded-full" style={{ left: drop < 0 ? `${50 - half}%` : "50%", width: `${half}%` }} />
    </div>
  )
}

export function RiskDrivers({ name, request, base, who }: { name: string; request: object; base: Risk; who: string }) {
  const [usual, setUsual] = useState<string[]>([])
  const [now, setNow] = useState<Risk>(base)
  const [busy, setBusy] = useState(false)
  const toggle = (feature: string) => {
    const next = usual.includes(feature) ? usual.filter((f) => f !== feature) : [...usual, feature]
    setUsual(next)
    setBusy(true)
    post<Risk>(`/forecasts/${name}/predict`, { ...request, usual: next })
      .then(setNow)
      .finally(() => setBusy(false))
  }
  const effects = base.effects ?? []
  const current = new Map((now.effects ?? []).map((e) => [e.feature, e]))
  const widest = Math.max(MIN_SCALE, ...effects.map((e) => Math.abs(e.drop)))
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="text-3xl font-semibold tabular-nums">{pct(now.chance ?? 0)}</span>
        {usual.length > 0 && <span className="text-muted-foreground text-sm tabular-nums">was {pct(base.chance ?? 0)}</span>}
        {busy && <Loader2 className="text-muted-foreground size-4 animate-spin" />}
        <span className="text-muted-foreground text-sm">{who}</span>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12" />
            <TableHead>Feature</TableHead>
            <TableHead>Value</TableHead>
            <TableHead className="w-1/4">
              <span className="flex items-center gap-1">
                Effect
                <InfoTip>How far this value moves the chance against the typical value. Switch a row off to score it with the typical value instead.</InfoTip>
              </span>
            </TableHead>
            <TableHead className="w-24 text-right">Points</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {effects.map((e) => {
            const on = !usual.includes(e.feature)
            const drop = current.get(e.feature)?.drop ?? 0
            return (
              <TableRow key={e.feature} className="cursor-pointer" onClick={() => toggle(e.feature)}>
                <TableCell>
                  <Switch checked={on} onClick={(ev) => ev.stopPropagation()} onCheckedChange={() => toggle(e.feature)} aria-label={e.feature} />
                </TableCell>
                <TableCell className="font-mono text-xs">{e.feature}</TableCell>
                <TableCell className={on ? "" : "text-muted-foreground"}>{featureValue(on ? e.value : e.typical)}</TableCell>
                <TableCell>
                  <EffectBar drop={drop} widest={widest} />
                </TableCell>
                <TableCell className="text-right tabular-nums">{points(drop)}</TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}
