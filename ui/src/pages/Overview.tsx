import { TrendingDown, TrendingUp } from "lucide-react"
import type { Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { rivalName, score, signed } from "@/format"
import { href } from "@/route"
import { leadOf, rivalScope } from "./specialist-groups"

const TIE = 0.0005

function Lead({ d }: { d: number }) {
  if (Math.abs(d) < TIE) return <Badge variant="outline">tie</Badge>
  return (
    <Badge variant={d > 0 ? "secondary" : "destructive"}>
      {d > 0 ? <TrendingUp /> : <TrendingDown />}
      {signed(d)}
    </Badge>
  )
}

export function Overview({ items }: { items: Summary[] }) {
  const rows = items
    .filter((s) => s.score != null && !!s.rival?.score)
    .map((s) => ({ s, d: leadOf(s) ?? 0 }))
    .sort((a, b) => b.d - a.d)
  const { shown, footer } = useShowMore(rows)
  if (!rows.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Against the best rival
          <InfoTip>Test score of each promoted model next to its strongest rival, the rival chosen on calib.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Ours</TableHead>
              <TableHead>Lead</TableHead>
              <TableHead>Best rival</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map(({ s, d }) => (
              <TableRow key={s.name} className="cursor-pointer" onClick={() => (window.location.hash = href("specialists", s.name))}>
                <TableCell className="font-medium">{s.title || s.name}</TableCell>
                <TableCell className="text-muted-foreground text-right tabular-nums">{score(s.score)}</TableCell>
                <TableCell>
                  <Lead d={d} />
                </TableCell>
                <TableCell className="text-muted-foreground/70">
                  <span className="block max-w-44 truncate">
                    <span className="tabular-nums">{score(s.rival?.score)}</span> {rivalName(s.rival?.name)}
                    {rivalScope(s)}
                  </span>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
      {footer}
    </Card>
  )
}
