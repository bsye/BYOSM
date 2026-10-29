import { ExternalLink } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { seconds, signed, signedInterval } from "@/format"
import { href } from "@/route"
import type { Effect } from "@/pages/EffectChart"

export type Status = "available" | "built-in" | "planned" | "lab"
type Outcome = { owner: string; method: string; result: string }
export type Technique = {
  id: string
  title: string
  effect: string
  status: Status
  modalities: string[]
  cost: number
  prior: number
  evidence: string[]
  evidence_dates?: (string | null)[]
  added?: string
  tries: number
  wins: number
  gpu_hours: number
  specialists: number
  calib_effect: Effect
  target_titles?: string[]
  outcomes?: Outcome[]
}

export const LAB: Status = "lab"
const STATUS_VARIANT: Record<Status, "default" | "secondary" | "outline"> = {
  available: "default",
  "built-in": "secondary",
  planned: "outline",
  lab: "outline",
}

const host = (url: string) => new URL(url).hostname.replace("www.", "")
const month = (ym: string | null) => (ym ? new Date(`${ym}-01T00:00:00`).toLocaleDateString(undefined, { month: "short", year: "numeric" }) : "")
const day = (iso: string) => new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })

export function Catalogue({ rows }: { rows: Technique[] }) {
  return (
    <Card>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Family</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Kinds</TableHead>
              <TableHead>Targets</TableHead>
              <TableHead className="text-right">Tries</TableHead>
              <TableHead className="text-right">Sure leads</TableHead>
              <TableHead className="text-right">Calib effect</TableHead>
              <TableHead className="text-right">GPU time</TableHead>
              <TableHead>Evidence</TableHead>
              <TableHead>Added</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((t) => (
              <TableRow key={t.id}>
                <TableCell className="max-w-96">
                  <div className="font-medium">{t.title}</div>
                  <div className="text-muted-foreground text-xs whitespace-normal">{t.effect}</div>
                  {(t.outcomes ?? []).map((o) => (
                    <a
                      key={`${o.owner}-${o.method}`}
                      href={href("specialists", o.owner, "lab")}
                      className="text-muted-foreground block text-xs hover:underline"
                    >
                      {o.owner} · {o.method}: {o.result}
                    </a>
                  ))}
                </TableCell>
                <TableCell>
                  <Badge variant={STATUS_VARIANT[t.status]}>{t.status}</Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">{t.modalities.join(", ") || "all"}</TableCell>
                <TableCell>
                  <div className="flex max-w-56 flex-wrap gap-1">
                    {(t.target_titles ?? []).map((c) => (
                      <Badge key={c} variant="outline" className="font-normal">
                        {c}
                      </Badge>
                    ))}
                  </div>
                </TableCell>
                <TableCell className="text-right tabular-nums">{t.tries}</TableCell>
                <TableCell className="text-right tabular-nums">{t.wins}</TableCell>
                <TableCell className="text-right tabular-nums">
                  <div>{signed(t.calib_effect.mean)}</div>
                  <div className="text-muted-foreground/70 text-xs">{signedInterval(t.calib_effect.lo, t.calib_effect.hi)}</div>
                </TableCell>
                <TableCell className="text-right tabular-nums">{t.gpu_hours ? seconds(t.gpu_hours * 3600) : "-"}</TableCell>
                <TableCell>
                  <div className="flex flex-col gap-1">
                    {t.evidence.map((url, i) => (
                      <a
                        key={url}
                        href={url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs"
                      >
                        {host(url)}
                        {t.evidence_dates?.[i] && <span className="text-muted-foreground/70">{month(t.evidence_dates[i])}</span>}
                        <ExternalLink className="size-3" />
                      </a>
                    ))}
                  </div>
                </TableCell>
                <TableCell className="text-muted-foreground text-xs whitespace-nowrap">{t.added ? day(t.added) : "-"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
