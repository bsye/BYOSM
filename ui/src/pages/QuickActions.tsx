import { Loader2, Play } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import { href } from "@/route"
import type { NextStep } from "./NextSteps"

function useSteps(items: Summary[]): Record<string, NextStep | null> {
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  const [steps, setSteps] = useState<Record<string, NextStep | null>>({})
  useEffect(() => {
    for (const s of items) {
      api<NextStep[]>(`/next-steps/${encodeURIComponent(s.name)}`).then(
        (found) => setSteps((cur) => ({ ...cur, [s.name]: found[0] ?? null })),
        () => setSteps((cur) => ({ ...cur, [s.name]: null })),
      )
    }
  }, [items, finished])
  return steps
}

function Step({ step }: { step: NextStep | null | undefined }) {
  if (step === undefined) return <Loader2 className="text-muted-foreground size-4 animate-spin" />
  if (!step) return null
  return (
    <span className="text-muted-foreground flex items-center gap-1">
      {step.title}
      <InfoTip>{step.why}</InfoTip>
    </span>
  )
}

export function QuickActions({ items }: { items: Summary[] }) {
  const steps = useSteps(items)
  const rows = items.filter((s) => steps[s.name] !== null)
  if (!rows.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quick actions</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Next step</TableHead>
              <TableHead className="text-right">Start</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((s) => {
              const step = steps[s.name]
              return (
                <TableRow key={s.name}>
                  <TableCell className="font-medium">
                    <a className="hover:underline" href={href("specialists", s.name)}>
                      {s.title || s.name}
                    </a>
                  </TableCell>
                  <TableCell>
                    <Step step={step} />
                  </TableCell>
                  <TableCell className="text-right">
                    <Button size="sm" variant="outline" disabled={!step} onClick={() => step && runJob(step.args).then((j) => openJob(j.id))}>
                      <Play />
                      Run
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
