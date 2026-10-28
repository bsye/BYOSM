import { CirclePause, RotateCw } from "lucide-react"
import { useState } from "react"
import { runJob, type Job } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { describe, retryArgs, unretried } from "@/jobs"
import { startAll } from "./stat-actions"

export function Interrupted({ jobs }: { jobs: Job[] }) {
  const lost = unretried(jobs, "interrupted")
  const [busy, setBusy] = useState(false)
  if (!lost.length) return null
  const retry = () => {
    setBusy(true)
    void startAll(lost.map((j) => runJob(retryArgs(j)))).finally(() => setBusy(false))
  }
  return (
    <Alert>
      <CirclePause />
      <AlertTitle>
        {lost.length} {lost.length === 1 ? "job was" : "jobs were"} interrupted when the app or the machine stopped
      </AlertTitle>
      <AlertDescription className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 truncate">{lost.map((j) => [describe(j).title, describe(j).target].filter(Boolean).join(" ")).join(", ")}</span>
        <Button size="sm" disabled={busy} onClick={retry}>
          <RotateCw />
          Retry all
        </Button>
      </AlertDescription>
    </Alert>
  )
}
