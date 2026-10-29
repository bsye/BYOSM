import { ArrowRight, Loader2, Play } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { useShowMore } from "@/components/show-more"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useEach } from "@/each"
import { useFinished } from "@/events"
import { openJob } from "@/job-sheet"
import { href } from "@/route"
import type { Healths } from "./HealthList"
import { MethodActions } from "./MethodActions"
import { runnable, type NextStep } from "./NextSteps"

type Pending = NextStep & { owner: string; approve: string }

export const best = (found: NextStep[]) => found.filter(runnable).sort((a, b) => (b.gain ?? 0) - (a.gain ?? 0))[0] ?? null

function usePending(): [Pending[], () => void] {
  const finished = useFinished()
  const [found, setFound] = useState<Pending[]>([])
  const load = useCallback(() => void api<Pending[]>("/lab/pending").then(setFound, () => setFound([])), [])
  useEffect(() => {
    load()
  }, [load, finished])
  return [found, load]
}

function useSteps(items: Summary[]): Record<string, NextStep | null> {
  const found = useEach<NextStep[]>(
    items.map((s) => s.name),
    "/next-steps",
    useFinished(),
  )
  return Object.fromEntries(Object.entries(found).map(([name, steps]) => [name, steps ? best(steps) : null]))
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
  const [pending, reload] = usePending()
  const titles = Object.fromEntries(items.map((s) => [s.name, s.title || s.name]))
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Next actions
          <InfoTip>
            Lab methods waiting for your approval first, then the step with the most health to gain for each model. Health shows the score now and at most after
            the step.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[38%]">Model</TableHead>
              <TableHead className="w-20 text-right">Health</TableHead>
              <TableHead>Next action</TableHead>
              <TableHead className="w-28" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {pending.map((a) => (
              <TableRow key={`${a.owner}-${a.id}`}>
                <TableCell className="truncate font-medium">
                  <a className="hover:underline" href={href("specialists", a.owner, "lab")}>
                    {titles[a.owner] ?? a.owner}
                  </a>
                </TableCell>
                <TableCell className="text-right">
                  <Change now={healths[a.owner]?.score} step={a} />
                </TableCell>
                <TableCell>
                  <Step step={a} />
                </TableCell>
                <TableCell className="text-right">
                  <MethodActions owner={a.owner} id={a.approve} status="pending" compact onDecided={reload} />
                </TableCell>
              </TableRow>
            ))}
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
