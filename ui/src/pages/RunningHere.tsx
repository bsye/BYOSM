import { Maximize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { UNFINISHED, useLive } from "@/events"
import { took } from "@/format"
import { openJob } from "@/job-sheet"
import { describe, stepNow } from "@/jobs"
import { useNow } from "@/now"
import { StatusIcon } from "./Jobs"

export function RunningHere({ name }: { name: string }) {
  const live = useLive()
  const jobs = Object.values(live.jobs).filter((j) => UNFINISHED.has(j.status) && describe(j).target === name)
  const now = useNow(jobs.length > 0)
  if (!jobs.length) return null
  return (
    <Card className="border-primary/40">
      <CardHeader>
        <CardTitle>Running now</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-2">
        {jobs.map((j) => (
          <div key={j.id} className="flex items-center gap-3 text-sm">
            <StatusIcon status={j.status} />
            <span className="font-medium">{describe(j).title}</span>
            <span className="text-muted-foreground truncate">{describe(j).subject}</span>
            <span className="text-muted-foreground min-w-0 flex-1 truncate text-xs">{stepNow(live.metrics[j.id] ?? [], live.lines[j.id] ?? [])}</span>
            <span className="text-muted-foreground tabular-nums">{j.started ? took(j.started, now) : "queued"}</span>
            <Button size="sm" variant="outline" onClick={() => openJob(j.id)}>
              <Maximize2 />
              Open
            </Button>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
