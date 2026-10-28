import { Check, Loader2, Play, X } from "lucide-react"
import { useState } from "react"
import { post, runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import { jobRunning } from "@/jobs"

export type MethodStatus = "pending" | "approved" | "rejected"
type Props = { owner: string; id: string; status: MethodStatus; problems?: string[]; ran?: boolean; compact?: boolean; onDecided?: () => void }

export function MethodActions({ owner, id, status, problems = [], ran = false, compact = false, onDecided }: Props) {
  const [decided, setDecided] = useState<MethodStatus | null>(null)
  const jobs = useLive().jobs
  const args = ["lab", "run", owner, id]
  const now = decided ?? status
  const decide = (next: MethodStatus) =>
    post(`/specialists/${owner}/lab/${id}`, { status: next }).then(() => {
      setDecided(next)
      onDecided?.()
    })
  if (now === "pending")
    return (
      <div className="flex justify-end gap-2">
        {!compact && (
          <Button size="sm" variant="outline" onClick={() => decide("rejected")}>
            <X />
            Reject
          </Button>
        )}
        <Button size="sm" disabled={problems.length > 0} onClick={() => decide("approved")}>
          <Check />
          Approve
        </Button>
      </div>
    )
  if (now !== "approved" || ran) return null
  const running = jobRunning(jobs, args)
  return (
    <Button size="sm" variant="outline" disabled={running} onClick={() => runJob(args).then((j) => openJob(j.id))}>
      {running ? <Loader2 className="animate-spin" /> : <Play />}
      Run
    </Button>
  )
}
