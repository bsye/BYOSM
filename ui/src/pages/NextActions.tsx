import { ArrowRight, Loader2, Play } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { useShowMore } from "@/components/show-more"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import { href } from "@/route"
import type { Healths } from "./HealthList"
import type { NextStep } from "./NextSteps"

export const best = (found: NextStep[]) => [...found].sort((a, b) => (b.gain ?? 0) - (a.gain ?? 0))[0] ?? null

function useSteps(items: Summary[]): Record<string, NextStep | null> {
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  const [steps, setSteps] = useState<Record<string, NextStep | null>>({})
  const names = items.map((s) => s.name).join("\n")
  useEffect(() => {
    for (const name of names.split("\n").filter(Boolean)) {
      api<NextStep[]>(`/next-steps/${encodeURIComponent(name)}`).then(
        (found) => setSteps((cur) => ({ ...cur, [name]: best(found) })),
        () => setSteps((cur) => ({ ...cur, [name]: null })),
      )
    }
  }, [names, finished])
  return steps
}

function Change({ now, step }: { now: number | undefined; step: NextStep | null | undefined }) {
  if (now == null) return <span className="text-muted-foreground">-</span>
  if (!step?.gain) return <span className="text-muted-foreground tabular-nums">{now}</span>
  return (
    <span className="flex items-center justify-end gap-1 tabular-nums">
      <span className="text-muted-foreground">{now}</span>
      <ArrowRight className="text-muted-foreground size-3" />
      <span className="font-medium">{Math.min(100, now + step.gain)}</span>
    </span>
  )
}

function Step({ step }: { step: NextStep | null | undefined }) {
  if (step === undefined) return <Loader2 className="text-muted-foreground size-4 animate-spin" />
  if (!step) return null
  return (
    <span className="text-muted-foreground flex min-w-0 items-center gap-1">
      <span className="truncate">{step.title}</span>
      <InfoTip>
        {step.why}
        {step.unblocks?.length ? ` Unblocks: ${step.unblocks.join(", ")}.` : ""}
      </InfoTip>
    </span>
  )
}

export function NextActions({ items, healths }: { items: Summary[]; healths: Healths }) {
  const { shown, footer } = useShowMore(items)
  const steps = useSteps(shown)
  const rows = shown.filter((s) => steps[s.name] !== null)
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Next actions
          <InfoTip>The step with the most health to gain for each model. Health shows the score now and at most after the step.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[38%]">Model</TableHead>
              <TableHead className="w-20 text-right">Health</TableHead>
              <TableHead>Next action</TableHead>
              <TableHead className="w-20" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((s) => {
              const step = steps[s.name]
              return (
                <TableRow key={s.name}>
                  <TableCell className="truncate font-medium">
                    <a className="hover:underline" href={href("specialists", s.name)}>
                      {s.title || s.name}
                    </a>
                  </TableCell>
                  <TableCell className="text-right">
                    <Change now={healths[s.name]?.score} step={step} />
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
      {footer}
    </Card>
  )
}
