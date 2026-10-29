import { UNFINISHED, useLive } from "@/events"
import { describe, stepNow } from "@/jobs"

export type Activity = { title: string; step: string }

export function useActivity(): Map<string, Activity> {
  const live = useLive()
  const active = Object.values(live.jobs)
    .filter((j) => UNFINISHED.has(j.status))
    .sort((a, b) => a.created - b.created)
  return new Map(active.map((j) => [describe(j).target ?? "", { title: describe(j).title, step: stepNow(live.metrics[j.id] ?? [], live.lines[j.id] ?? []) }]))
}
