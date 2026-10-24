import { Loader2, Repeat } from "lucide-react"
import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api, runJob, type Detail } from "@/api"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"

type Plan = { from: string | null; epochs?: number; reasons: string[] }

export function ContinueDialog({ d }: { d: Detail }) {
  const [plan, setPlan] = useState<Plan | null>(null)
  const [starting, setStarting] = useState(false)
  const running = Object.values(useLive().jobs).some(
    (j) => j.args[0] === "continue" && j.args[1] === d.name && (j.status === "running" || j.status === "queued"),
  )
  useEffect(() => {
    api<Plan>(`/specialists/${d.name}/continue-plan`).then(setPlan, () => setPlan(null))
  }, [d.name, d.promoted])
  const go = () => {
    setStarting(true)
    runJob(["continue", d.name, "--run"])
      .then(
        (j) => openJob(j.id),
        (e) => toast.error(String(e)),
      )
      .finally(() => setStarting(false))
  }
  const busy = running || starting
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          <Button variant="outline" disabled={!plan?.from || busy} onClick={go}>
            {busy ? <Loader2 className="animate-spin" /> : <Repeat />}
            {running ? "Training" : "Keep training"}
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-sm">
        {plan?.from ? (
          <div className="grid gap-1">
            <span className="font-medium">From {plan.from}, based on its health and dev errors:</span>
            {plan.reasons.map((r) => (
              <span key={r}>· {r}</span>
            ))}
            <span>· rows aimed at its errors are written before training</span>
          </div>
        ) : (
          "Train it first"
        )}
      </TooltipContent>
    </Tooltip>
  )
}
