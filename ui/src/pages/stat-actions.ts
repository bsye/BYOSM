import { toast } from "sonner"
import { api, runJob } from "@/api"
import { openJob } from "@/job-sheet"
import type { NextStep } from "./NextSteps"
import { best } from "./NextActions"

type Llm = Record<string, unknown> | null
export type Progress = (started: number, total: number) => void

const counted = <T>(runs: Promise<T>[], progress?: Progress): Promise<T>[] => {
  let n = 0
  progress?.(0, runs.length)
  return runs.map((r) =>
    r.finally(() => {
      n += 1
      progress?.(n, runs.length)
    }),
  )
}

const steps = (name: string) => api<NextStep[]>(`/next-steps/${encodeURIComponent(name)}`)

export async function startAll(runs: Promise<{ id: string } | null>[], progress?: Progress) {
  const done = await Promise.allSettled(counted(runs, progress))
  const jobs = done.flatMap((r) => (r.status === "fulfilled" && r.value ? [r.value] : []))
  const failed = done.flatMap((r) => (r.status === "rejected" ? [String(r.reason)] : []))
  if (failed.length) toast.error(`${failed.length} could not start`, { description: failed[0] })
  if (jobs[0]) openJob(jobs[0].id)
}

export const untilReady = (names: string[], llm: Llm, progress?: Progress) =>
  startAll(
    names.map((n) => runJob(["ready", n], llm)),
    progress,
  )

export const bestSteps = (names: string[], llm: Llm, progress?: Progress) =>
  startAll(
    names.map((n) =>
      steps(n).then((found) => {
        const step = best(found)
        return step ? runJob(step.args, llm) : null
      }),
    ),
    progress,
  )

export const unblock = (names: string[], check: string, llm: Llm, progress?: Progress) =>
  startAll(
    names.map((n) =>
      steps(n).then((found) => {
        const step = found.find((s) => s.unblocks?.includes(check))
        return step ? runJob(step.args, llm) : null
      }),
    ),
    progress,
  )
