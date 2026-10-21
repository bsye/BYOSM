import type { Job } from "@/api"
import type { Metric } from "@/events"

const VERBS: Record<string, string> = {
  auto: "Create",
  build: "Build",
  eval: "Score",
  improve: "Improve",
  export: "Export",
  data: "Build data",
  distill: "Distill",
  "check-data": "Check data",
  "request-data": "Request data",
  promote: "Promote",
  report: "Report",
  board: "Board",
  show: "Show",
  ls: "List",
  new: "Define",
  exp: "Experiment",
  suite: "Suite",
  baseline: "Rival",
  overfit: "Overfitting check",
}

export type Described = { title: string; target: string | null; subject: string | null }

export function describe(job: Pick<Job, "args">): Described {
  const [cmd, ...rest] = job.args
  if (cmd === "system") return { title: `${rest[0] === "eval" ? "Score" : "Build"} team`, target: rest[1] ?? null, subject: null }
  const verb = cmd === "improve" && rest.includes("--apply") ? "Run proposal" : (VERBS[cmd] ?? cmd)
  const plain = rest.filter((a) => !a.startsWith("--"))
  if (cmd === "auto" && plain.length === 1 && plain[0].includes(" ")) return { title: verb, target: null, subject: plain[0] }
  return { title: verb, target: plain[0] ?? null, subject: plain.slice(1).join(" ") || null }
}

export function currentStep(metrics: Metric[], lines: string[]): string {
  const last = metrics.at(-1)
  if (last?.kind === "llm" && last.state === "waiting") return `Asking ${modelName(String(last.model))} (${last.role})`
  if (last?.kind === "epoch") return `Training ${last.experiment}, epoch ${last.epoch}/${last.epochs}`
  if (last?.kind === "teacher") return `${last.teacher} labelling ${last.done}/${last.total}`
  if (last?.kind === "scored") return `Scored ${last.predictor} on ${last.suite}`
  return lines.at(-1)?.trim() || "Starting"
}

const PROVIDER_PREFIXES = ["openai:", "anthropic:", "claude-code:", "codex:"]

export function modelName(id: string): string {
  const at = id.indexOf("@")
  const bare = at >= 0 ? id.slice(0, at) : id
  const prefix = PROVIDER_PREFIXES.find((p) => bare.startsWith(p))
  return prefix ? bare.slice(prefix.length) : bare
}

export function llmUsed(metrics: Metric[]): string | null {
  const m = metrics.find((x) => x.kind === "llm")
  return m ? modelName(String(m.model)) : null
}

export type Step = { label: string; detail: string; state: "running" | "done" | "failed" }

export function steps(metrics: Metric[], running: boolean): Step[] {
  const out: Step[] = []
  const calls = new Map<string, Metric>()
  for (const m of metrics) {
    if (m.kind === "llm") calls.set(String(m.call), { ...(calls.get(String(m.call)) ?? {}), ...m })
  }
  const seen = new Set<string>()
  for (const m of metrics) {
    if (m.kind === "llm") {
      const id = String(m.call)
      if (seen.has(id)) continue
      seen.add(id)
      const c = calls.get(id) as Metric
      out.push({
        label: `Ask ${c.role}`,
        detail: `${c.model}${c.seconds != null ? `, ${c.seconds} s` : ""}`,
        state: c.state === "waiting" ? "running" : c.state === "failed" ? "failed" : "done",
      })
    } else if (m.kind === "epoch") {
      const prev = out.at(-1)
      const step = { label: `Train ${m.experiment}`, detail: `epoch ${m.epoch}/${m.epochs}, dev ${Number(m.dev_objective).toFixed(3)}`, state: "done" as const }
      if (prev?.label === step.label) out[out.length - 1] = step
      else out.push(step)
    } else if (m.kind === "teacher") {
      const prev = out.at(-1)
      const step = { label: `Label with ${m.teacher}`, detail: `${m.done}/${m.total}`, state: "done" as const }
      if (prev?.label === step.label) out[out.length - 1] = step
      else out.push(step)
    } else if (m.kind === "scored") {
      out.push({ label: `Score ${m.predictor}`, detail: `${m.suite}: ${m.objective == null ? "-" : Number(m.objective).toFixed(3)}`, state: "done" })
    }
  }
  if (running && out.length && out.at(-1)?.state === "done" && metrics.at(-1)?.kind !== "scored") out[out.length - 1] = { ...(out.at(-1) as Step), state: "running" }
  return out
}
