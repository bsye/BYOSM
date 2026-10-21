import { ChevronDown, ChevronRight, MessagesSquare } from "lucide-react"
import { Fragment, useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"

const TICK_MS = 1000
const SHOWN = 200

type Call = {
  call: string
  job: string
  command: string
  role: string
  model: string
  state: "waiting" | "done" | "failed"
  started: number
  seconds?: number
  prompt?: string
  answer?: string
  error?: string
}

const STATE_VARIANT = { waiting: "default", done: "secondary", failed: "destructive" } as const

function useCalls(): Call[] {
  const live = useLive()
  const calls = new Map<string, Call>()
  for (const [job, events] of Object.entries(live.metrics)) {
    const command = (live.jobs[job]?.args ?? []).slice(0, 2).join(" ")
    for (const e of events) {
      if (e.kind !== "llm") continue
      const id = String(e.call ?? "")
      const before = calls.get(id)
      calls.set(id, {
        ...(before ?? { started: Number(e.at ?? Date.now() / 1000) }),
        ...(e as unknown as Partial<Call>),
        call: id,
        job,
        command,
      } as Call)
    }
  }
  return [...calls.values()].reverse().slice(0, SHOWN)
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now() / 1000), TICK_MS)
    return () => clearInterval(t)
  }, [active])
  return now
}

export function LlmInspector() {
  const calls = useCalls()
  const waiting = calls.filter((c) => c.state === "waiting").length
  const now = useNow(waiting > 0)
  const [open, setOpen] = useState<string | null>(null)
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" className="fixed right-4 bottom-4 z-40 shadow-xs">
          <MessagesSquare />
          LLM calls
          {calls.length > 0 && <Badge variant={waiting ? "default" : "secondary"}>{waiting ? `${waiting} waiting` : calls.length}</Badge>}
        </Button>
      </SheetTrigger>
      <SheetContent side="bottom" className="max-h-[70vh]">
        <SheetHeader>
          <SheetTitle>LLM calls</SheetTitle>
          <SheetDescription>
            Every call a running job makes to its LLM: which role asked, which model answered, how long it took, and what was
            sent and returned.
          </SheetDescription>
        </SheetHeader>
        <div className="overflow-auto px-4 pb-4">
          {calls.length === 0 ? (
            <p className="text-muted-foreground text-sm">No LLM calls since this page opened. They appear here as jobs make them.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8" />
                  <TableHead>Role</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Job</TableHead>
                  <TableHead>State</TableHead>
                  <TableHead className="text-right">Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {calls.map((c) => {
                  const expanded = open === c.call
                  const secs = c.seconds ?? Math.max(0, now - c.started)
                  return (
                    <Fragment key={c.call}>
                      <TableRow className="cursor-pointer" onClick={() => setOpen(expanded ? null : c.call)}>
                        <TableCell>{expanded ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}</TableCell>
                        <TableCell className="font-medium">{c.role}</TableCell>
                        <TableCell className="text-foreground/70">{c.model}</TableCell>
                        <TableCell className="text-muted-foreground font-mono text-xs">byosm {c.command}</TableCell>
                        <TableCell>
                          <Badge variant={STATE_VARIANT[c.state]}>{c.state}</Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{secs.toFixed(1)} s</TableCell>
                      </TableRow>
                      {expanded && (
                        <TableRow>
                          <TableCell />
                          <TableCell colSpan={5} className="space-y-3 whitespace-pre-wrap">
                            <div>
                              <p className="text-muted-foreground text-xs">Prompt</p>
                              <p className="font-mono text-xs">{c.prompt || "-"}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground text-xs">{c.error ? "Error" : "Answer"}</p>
                              <p className="font-mono text-xs">{c.error || c.answer || "waiting..."}</p>
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}
