import type { Metric } from "@/events"
import type { Step } from "@/jobs"

export type StageState = "running" | "done" | "failed"
export type Stage = { name: string; state: StageState; start: number; seconds: number | null; steps: Step[] }
export type Alternative = { option: string; why_not: string; args?: string[] }
export type Decision = { step: string; choice: unknown; why: string; by?: string; alternatives: Alternative[] }

const QUESTIONS: Record<string, string> = {
  define: "What is the task?",
  "discover:benchmark": "What data to learn from and test on?",
  "discover:data": "Write training data with the LLM?",
  "discover:baseline": "Which rival to beat?",
  data: "Is there enough data?",
  build: "How does the first model score?",
  stop: "Keep improving?",
  distill: "Make it smaller?",
  overfit: "Is it overfitting?",
  export: "Export it?",
  publish: "Publish it?",
}

export const question = (step: string) => QUESTIONS[step] ?? step.split(":").at(-1) ?? step

export function stages(metrics: Metric[], steps: Step[]): Stage[] {
  const out: Stage[] = []
  for (const m of metrics.filter((x) => x.kind === "stage")) {
    const name = String(m.name)
    if (m.state === "running") {
      out.push({ name, state: "running", start: Number(m.at), seconds: null, steps: [] })
      continue
    }
    const open = out.findLast((s) => s.name === name && s.state === "running")
    const seconds = Number(m.seconds)
    if (open) Object.assign(open, { state: m.state, seconds })
    else out.push({ name, state: m.state as StageState, start: Number(m.at) - seconds, seconds, steps: [] })
  }
  for (const step of steps) out.findLast((s) => (step.at ?? 0) >= s.start)?.steps.push(step)
  return out
}

export function decisions(metrics: Metric[]): Decision[] {
  return metrics
    .filter((m) => m.kind === "decision")
    .map((m) => ({
      step: String(m.step),
      choice: m.choice,
      why: String(m.why ?? ""),
      by: m.by ? String(m.by) : undefined,
      alternatives: (m.alternatives as Alternative[] | undefined) ?? [],
    }))
}
