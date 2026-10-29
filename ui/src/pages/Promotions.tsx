import type { Promotion } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { stamp } from "@/format"
import { describe } from "@/jobs"
import { href } from "@/route"

export function Promotions({ items }: { items: Promotion[] }) {
  const jobs = useLive().jobs
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Promotions
          <InfoTip>Every change of the shipped model, newest first: what it replaced, why, and the job that made it.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>When</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Why</TableHead>
              <TableHead>By</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.toReversed().map((r) => (
              <TableRow key={`${r.at}-${r.after}`}>
                <TableCell className="text-foreground/70 text-xs whitespace-nowrap tabular-nums">{stamp(r.at)}</TableCell>
                <TableCell className="whitespace-nowrap">
                  <span className="text-muted-foreground">{r.before ?? "none"} to </span>
                  <span className="font-medium">{r.after}</span>
                </TableCell>
                <TableCell className="text-muted-foreground max-w-md text-xs whitespace-normal">{r.why}</TableCell>
                <TableCell className="text-xs">
                  {jobs[r.by] ? (
                    <a className="hover:underline" href={href("jobs", r.by)}>
                      {describe(jobs[r.by]).title}
                    </a>
                  ) : (
                    r.by
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
