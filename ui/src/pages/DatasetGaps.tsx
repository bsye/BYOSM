import { Hammer, Loader2, RefreshCw } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { compact } from "@/format"
import { NewSpecialist } from "./NewSpecialist"

type Gap = { task: string; language: string | null; demand: number; supply: number; datasets: number; score: number; top_model: string | null; prompt: string }
type Gaps = { at: number | null; rows: Gap[] }

const SHOWN = 20

function useGaps(): [Gaps | null, boolean] {
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "dataset-gaps")
  const running = jobs.some((j) => j.status === "running" || j.status === "queued")
  const done = jobs.filter((j) => j.status === "done").length
  const [gaps, setGaps] = useState<Gaps | null>(null)
  useEffect(() => {
    api<Gaps>("/datasets/gaps").then(setGaps, () => setGaps({ at: null, rows: [] }))
  }, [done])
  return [gaps, running]
}

export function DatasetGaps() {
  const [gaps, running] = useGaps()
  const scan = () => void runJob(["dataset-gaps"])
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Gaps on Hugging Face
          <InfoTip>
            Tasks and languages whose top 50 models are downloaded far more than their top 50 datasets. Hugging Face does not publish search volumes, so
            downloads stand in for demand. Build turns a gap into a specialist, and its training data into a dataset to publish.
          </InfoTip>
        </CardTitle>
        <CardAction>
          <Button size="sm" variant="outline" onClick={scan}>
            {running ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            {running ? "Scanning" : "Scan"}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {!gaps ? (
          <Skeleton className="h-48 rounded-lg" />
        ) : !gaps.rows.length ? (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>Not scanned yet</EmptyTitle>
              <EmptyDescription>Scan ranks where datasets are missing.</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Gap</TableHead>
                <TableHead className="text-right">Model downloads</TableHead>
                <TableHead className="text-right">Dataset downloads</TableHead>
                <TableHead className="text-right">Ratio</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {gaps.rows.slice(0, SHOWN).map((g) => (
                <TableRow key={g.prompt}>
                  <TableCell>
                    <div className="font-medium">{g.prompt.charAt(0).toUpperCase() + g.prompt.slice(1)}</div>
                    {g.top_model && <div className="text-muted-foreground text-xs">top model {g.top_model}</div>}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{compact(g.demand)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {compact(g.supply)} <span className="text-muted-foreground">({g.datasets})</span>
                  </TableCell>
                  <TableCell className="text-right font-medium tabular-nums">{g.score.toFixed(0)}x</TableCell>
                  <TableCell className="text-right">
                    <NewSpecialist initial={g.prompt}>
                      <Button size="sm" variant="ghost">
                        <Hammer />
                        Build
                      </Button>
                    </NewSpecialist>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
