import { toast } from "sonner"
import { api, runJob } from "@/api"
import { openJob } from "@/job-sheet"
import type { NextStep } from "./NextSteps"
import { best } from "./NextActions"

type Llm = Record<string, unknown> | null

const steps = (name: string) => api<NextStep[]>(`/next-steps/${encodeURIComponent(name)}`)

export async function startAll(runs: Promise<{ id: string } | null>[]) {
  const done = await Promise.allSettled(runs)
  const jobs = done.flatMap((r) => (r.status === "fulfilled" && r.value ? [r.value] : []))
  const failed = done.flatMap((r) => (r.status === "rejected" ? [String(r.reason)] : []))
  if (failed.length) toast.error(`${failed.length} could not start`, { description: failed[0] })
  if (jobs[0]) openJob(jobs[0].id)
}

export const untilReady = (names: string[], llm: Llm) => startAll(names.map((n) => runJob(["ready", n], llm)))

export const bestSteps = (names: string[], llm: Llm) =>
  startAll(
    names.map((n) =>
      steps(n).then((found) => {
        const step = best(found)
        return step ? runJob(step.args, llm) : null
      }),
    ),
  )

export const unblock = (names: string[], check: string, llm: Llm) =>
  startAll(
    names.map((n) =>
      steps(n).then((found) => {
        const step = found.find((s) => s.unblocks?.includes(check))
        return step ? runJob(step.args, llm) : null
      }),
    ),
  )
