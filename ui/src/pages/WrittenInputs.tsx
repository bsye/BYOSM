import { Loader2 } from "lucide-react"
import { useEffect, useState } from "react"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"

type Written = { suite: string; frozen: boolean; rows: { input: string; label: string }[] }

const UNFINISHED = new Set(["running", "queued"])

function suiteOf(args: string[]): string | null {
  const at = args.indexOf("--suite")
  return at >= 0 ? (args[at + 1] ?? null) : null
}

function useWritten(name: string): { logs: Written[]; writing: string | null } {
  const live = useLive()
  const job = Object.values(live.jobs).find((j) => j.args[0] === "testset" && j.args[1] === name && UNFINISHED.has(j.status))
  const calls = job ? (live.metrics[job.id] ?? []).filter((m) => m.kind === "writing").length : 0
  const ended = Object.values(live.jobs).filter((j) => j.args[0] === "testset" && !UNFINISHED.has(j.status)).length
  const [logs, setLogs] = useState<Written[]>([])
  useEffect(() => {
    api<Written[]>(`/specialists/${name}/written`)
      .then(setLogs)
      .catch(() => setLogs([]))
  }, [name, calls, ended])
  return { logs, writing: job ? suiteOf(job.args) : null }
}

function State({ log, writing }: { log: Written; writing: string | null }) {
  if (log.suite === writing)
    return (
      <Badge variant="secondary">
        <Loader2 className="animate-spin" />
        Writing
      </Badge>
    )
  return log.frozen ? <Badge variant="outline">Scored</Badge> : <Badge variant="outline">Stopped</Badge>
}

export function WrittenInputs({ name }: { name: string }) {
  const { logs, writing } = useWritten(name)
  const [picked, setPicked] = useState<string | null>(null)
  const log = logs.find((l) => l.suite === (picked ?? writing)) ?? logs[0]
  if (!log) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {log.rows.length} inputs written
          <State log={log} writing={writing} />
          <InfoTip>Every input the LLM wrote, newest first, kept after scoring. A stopped run picks up from here next time.</InfoTip>
        </CardTitle>
        {logs.length > 1 && (
          <CardAction>
            <Select value={log.suite} onValueChange={setPicked}>
              <SelectTrigger size="sm" aria-label="Written set">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {logs.map((l) => (
                  <SelectItem key={l.suite} value={l.suite}>
                    {l.suite}, {l.rows.length}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardAction>
        )}
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-80">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Input</TableHead>
                <TableHead>Label</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...log.rows].reverse().map((r, i) => (
                <TableRow key={log.rows.length - i}>
                  <TableCell className="whitespace-normal">{r.input}</TableCell>
                  <TableCell className="text-foreground/70">{r.label}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
