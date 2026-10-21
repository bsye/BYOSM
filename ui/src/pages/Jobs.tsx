import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { took, when } from "@/format"
import { href } from "@/route"
import { JobLog, StatusBadge } from "./JobLog"

export function Running() {
  const live = useLive()
  const active = Object.values(live.jobs).filter((j) => j.status === "running" || j.status === "queued")
  if (!active.length) return null
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      {active.map((j) => {
        const last = (live.lines[j.id] ?? []).at(-1) ?? "waiting for output..."
        return (
          <a key={j.id} href={href("jobs", j.id)}>
            <Card className="hover:bg-accent/50 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
                <CardTitle className="truncate font-mono text-sm">byosm {j.args.join(" ")}</CardTitle>
                <StatusBadge status={j.status} />
              </CardHeader>
              <CardContent className="text-muted-foreground truncate font-mono text-xs">{last}</CardContent>
            </Card>
          </a>
        )
      })}
    </div>
  )
}

export function Jobs({ id }: { id?: string }) {
  const jobs = Object.values(useLive().jobs).sort((a, b) => b.created - a.created)
  return (
    <div className="space-y-4">
      {id && <JobLog id={id} />}
      <Card>
        <CardHeader>
          <CardTitle>Jobs</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>command</TableHead>
                <TableHead>status</TableHead>
                <TableHead>started</TableHead>
                <TableHead>took</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((j) => (
                <TableRow key={j.id} data-state={j.id === id ? "selected" : undefined}>
                  <TableCell className="font-mono text-xs">
                    <a className="hover:underline" href={href("jobs", j.id)}>
                      byosm {j.args.join(" ")}
                    </a>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={j.status} />
                  </TableCell>
                  <TableCell>{when(j.started)}</TableCell>
                  <TableCell>{took(j.started, j.ended)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
