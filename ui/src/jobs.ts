import type { Job } from "@/api"
import type { Metric } from "@/events"
import { seconds } from "@/format"

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
  testset: "Write test set",
  "adopt-data": "Adopt data",
  next: "Do next steps",
  datasets: "Find datasets",
}

const INPUT_VERBS: Record<string, string> = { write: "Write inputs", sample: "Sample inputs", run: "Score inputs" }
const SUB_VERBS: Record<string, Record<string, string>> = {
  baseline: { add: "Add rival" },
  suite: { add: "Add suite" },
  exp: { create: "Create experiment" },
}
const NAMED_FLAGS = new Set(["--apply", "--suite"])
export const EVENT_PREFIX = "@event "

export type Described = { title: string; target: string | null; subject: string | null }

export function describe(job: Pick<Job, "args">): Described {
  const [cmd, ...rest] = job.args
  const sub = SUB_VERBS[cmd]?.[rest[0]]
  if (sub) return { title: sub, target: rest[1] ?? null, subject: rest[2] ?? null }
  if (cmd === "inputs") return { title: INPUT_VERBS[rest[0]] ?? "Test inputs", target: rest[1] ?? null, subject: rest[rest.indexOf("--set") + 1] ?? null }
  if (cmd === "system") return { title: `${rest[0] === "eval" ? "Score" : "Build"} team`, target: rest[1] ?? null, subject: null }
  const verb = cmd === "improve" && rest.includes("--apply") ? "Run proposal" : (VERBS[cmd] ?? cmd)
  const plain = rest.filter((a, i) => !a.startsWith("--") && !(rest[i - 1]?.startsWith("--") && !NAMED_FLAGS.has(rest[i - 1])))
  if (cmd === "auto" && plain.length === 1 && plain[0].includes(" ")) return { title: verb, target: null, subject: plain[0] }
  return { title: verb, target: plain[0] ?? null, subject: plain.slice(1).join(" ") || null }
}

const ROLE_TASKS: Record<string, string> = {
  spec: "draft the spec",
  pick: "pick datasets",
  map: "map dataset columns",
  generate: "write training data",
  eval: "write test data",
  judge: "check labels",
  analyze: "study the errors",
  baseline: "answer as a rival",
}

export const roleTask = (role: unknown) => ROLE_TASKS[String(role)] ?? String(role)

function openStage(metrics: Metric[]): string | null {
  const open: string[] = []
  for (const m of metrics) {
    if (m.kind !== "stage") continue
    if (m.state === "running") open.push(String(m.name))
    else open.splice(open.lastIndexOf(String(m.name)), 1)
  }
  return open.at(-1) ?? null
}

export function currentStep(metrics: Metric[], lines: string[]): string {
  const last = metrics.at(-1)
  if (last?.kind === "llm") return `${modelName(String(last.model))}: ${roleTask(last.role)}`
  if (last?.kind === "epoch") return `Training ${last.experiment}, epoch ${last.epoch}/${last.epochs}`
  if (last?.kind === "teacher") return `${last.teacher} labelling ${last.done}/${last.total}`
  if (last?.kind === "writing") return `Writing ${last.split} data ${last.done}/${last.total}`
  if (last?.kind === "scored") return `Scored ${last.predictor} on ${last.suite}`
  return openStage(metrics) ?? (lines.findLast((l) => l.trim() && !l.startsWith(EVENT_PREFIX))?.trim() || "Starting")
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

export type Step = { label: string; detail: string; state: "running" | "done" | "failed"; at?: number }

type CallGroup = { label: string; calls: number; failed: number; waiting: number; start: number; end: number }

function callStep(g: CallGroup): Step {
  const span = g.end - g.start
  const time = span > 0 ? `, ${seconds(span)}` : ""
  const failed = g.failed ? `, ${g.failed} failed` : ""
  const detail = g.calls === 1 ? (g.waiting ? "waiting" : seconds(span)) : `${g.calls} calls${time}${failed}`
  return { label: g.label, detail, state: g.waiting ? "running" : g.failed === g.calls ? "failed" : "done", at: g.start }
}

function progressStep(m: Metric): Step | null {
  if (m.kind === "epoch") return { label: `Train ${m.experiment}`, detail: `epoch ${m.epoch}/${m.epochs}, dev ${Number(m.dev_objective).toFixed(3)}`, state: "done" }
  if (m.kind === "teacher") return { label: `Label with ${m.teacher}`, detail: `${m.done}/${m.total}`, state: "done" }
  if (m.kind === "scored") return { label: `Score ${m.predictor}`, detail: `${m.suite}: ${m.objective == null ? "-" : Number(m.objective).toFixed(3)}`, state: "done" }
  return null
}

export function steps(metrics: Metric[], running: boolean): Step[] {
  const out: Step[] = []
  const calls = new Map<string, Metric>()
  for (const m of metrics) {
    if (m.kind === "llm") calls.set(String(m.call), { ...(calls.get(String(m.call)) ?? {}), ...m })
  }
  const seen = new Set<string>()
  let group = null as CallGroup | null
  for (const m of metrics) {
    if (m.kind === "llm") {
      const id = String(m.call)
      if (seen.has(id)) continue
      seen.add(id)
      const c = calls.get(id) as Metric
      const label = `${modelName(String(c.model))}: ${roleTask(c.role)}`
      if (group?.label !== label || out.at(-1)?.label !== label) {
        group = { label, calls: 0, failed: 0, waiting: 0, start: Number(c.at), end: Number(c.at) }
        out.push(callStep(group))
      }
      group.calls += 1
      group.failed += c.state === "failed" ? 1 : 0
      group.waiting += c.state === "waiting" ? 1 : 0
      group.end = Math.max(group.end, Number(c.at) + Number(c.waited ?? 0) + Number(c.seconds ?? 0))
      out[out.length - 1] = callStep(group)
      continue
    }
    const found = progressStep(m)
    if (!found) continue
    const step = { ...found, at: Number(m.at) }
    if (m.kind !== "scored" && out.at(-1)?.label === step.label) out[out.length - 1] = step
    else out.push(step)
  }
  if (running && out.length && out.at(-1)?.state === "done" && metrics.at(-1)?.kind !== "scored") out[out.length - 1] = { ...(out.at(-1) as Step), state: "running" }
  return out
}

const PROGRESS_KINDS: Record<string, { done: string; total: string }> = {
  writing: { done: "done", total: "total" },
  teacher: { done: "done", total: "total" },
  epoch: { done: "epoch", total: "epochs" },
}

export function timeLeft(metrics: Metric[]): string | null {
  const last = metrics.at(-1)
  const keys = last && PROGRESS_KINDS[last.kind]
  if (!last || !keys) return null
  const same = metrics.filter((m) => m.kind === last.kind && m.at != null && Object.keys(m).every((k) => k === keys.done || k === "at" || m[k] === last[k] || !(k in last)))
  const first = same[0]
  const done = Number(last[keys.done]) - Number(first?.[keys.done] ?? 0)
  const total = Number(last[keys.total])
  if (!first || done <= 0 || first === last) return null
  const perStep = (Number(last.at) - Number(first.at)) / done
  const left = perStep * (total - Number(last[keys.done]))
  return left > 0 ? `${seconds(left)} left` : null
}
