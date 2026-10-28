import type { Isolation } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

function limits(i: Isolation): string {
  const parts = [i.cpus != null && `${i.cpus} CPUs`, i.memory_gb != null && `${i.memory_gb} GB`, i.gpus && `GPUs ${i.gpus}`, i.nice != null && `nice ${i.nice}`]
  return parts.filter(Boolean).join(", ")
}

export function IsolationBadge({ isolation }: { isolation?: Isolation | null }) {
  if (!isolation || (isolation.mode === "process" && !isolation.fallback)) return null
  const label = isolation.mode === "container" ? "Container" : "Process"
  const why = isolation.mode === "container" ? limits(isolation) : isolation.fallback
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
          {label}
        </Badge>
      </TooltipTrigger>
      <TooltipContent>{why}</TooltipContent>
    </Tooltip>
  )
}
