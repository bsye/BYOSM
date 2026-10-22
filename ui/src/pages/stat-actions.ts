import { api, runJob } from "@/api"
import { openJob } from "@/job-sheet"
import type { NextStep } from "./NextSteps"
import { best } from "./NextActions"

type Llm = Record<string, unknown> | null

const steps = (name: string) => api<NextStep[]>(`/next-steps/${encodeURIComponent(name)}`)

async function startAll(runs: Promise<{ id: string } | null>[]) {
  const jobs = (await Promise.all(runs)).filter((j): j is { id: string } => j != null)
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
