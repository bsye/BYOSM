import { Loader2, Rocket } from "lucide-react"
import { useEffect, useState } from "react"
import { api, type Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { href } from "@/route"
import { untilReady } from "./stat-actions"

export type Health = { score: number; ready: boolean; ready_score: number; blockers: string[] }
export type Healths = Record<string, Health | null>

export function useHealths(items: Summary[] | null): Healths {
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  const [found, setFound] = useState<Healths>({})
  useEffect(() => {
    for (const s of items ?? []) {
      api<Health | null>(`/health/${encodeURIComponent(s.name)}`).then(
        (h) => setFound((cur) => ({ ...cur, [s.name]: h })),
        () => setFound((cur) => ({ ...cur, [s.name]: null })),
      )
    }
  }, [items, finished])
  return found
}

function Status({ h }: { h: Health | null | undefined }) {
  if (h === undefined) return <Loader2 className="text-muted-foreground size-4 animate-spin" />
  if (!h) return <span className="text-muted-foreground">-</span>
  if (h.ready) return <Badge>Ready</Badge>
  return <Badge variant="outline">{h.score >= h.ready_score ? "Blocked" : `Below ${h.ready_score}`}</Badge>
}

function AutoReady({ names, done }: { names: string[]; done: () => void }) {
  const start = () => untilReady(names, null).then(done)
  return (
    <Button size="sm" disabled={!names.length} onClick={start}>
      <Rocket />
      Auto until ready{names.length ? ` (${names.length})` : ""}
    </Button>
  )
}

export function HealthList({ items, healths }: { items: Summary[]; healths: Healths }) {
  const [picked, setPicked] = useState<string[]>([])
  const score = (s: Summary) => healths[s.name]?.score ?? -1
  const rows = items.filter((s) => healths[s.name] !== null).sort((a, b) => score(b) - score(a))
  const { shown, footer } = useShowMore(rows)
  const toggle = (name: string, on: boolean) => setPicked((cur) => (on ? [...cur, name] : cur.filter((n) => n !== name)))
  if (!rows.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Health
          <InfoTip>Weighted checks on each promoted model. Ready for production at 80 or more with no failing check. Auto runs each model's next steps until it is ready.</InfoTip>
        </CardTitle>
        <CardAction>
          <AutoReady names={picked} done={() => setPicked([])} />
        </CardAction>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8" />
              <TableHead>Model</TableHead>
              <TableHead>Score</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Blocked by</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((s) => {
              const h = healths[s.name]
              return (
                <TableRow
                  key={s.name}
                  className="cursor-pointer"
                  data-state={picked.includes(s.name) ? "selected" : undefined}
                  onClick={() => (window.location.hash = href("specialists", s.name))}
                >
                  <TableCell onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      aria-label={`Auto ${s.title || s.name}`}
                      disabled={!h || h.ready}
                      checked={picked.includes(s.name)}
                      onCheckedChange={(on) => toggle(s.name, on === true)}
                    />
                  </TableCell>
                  <TableCell className="font-medium">{s.title || s.name}</TableCell>
                  <TableCell>
                    {h && (
                      <span className="flex items-center gap-2">
                        <Progress value={h.score} className="w-16" />
                        <span className="text-muted-foreground tabular-nums">{h.score}</span>
                      </span>
                    )}
                  </TableCell>
                  <TableCell>
                    <Status h={h} />
                  </TableCell>
                  <TableCell className="text-muted-foreground/70">
                    <span className="block max-w-44 truncate">{h?.blockers.join(", ")}</span>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
      {footer}
    </Card>
  )
}
