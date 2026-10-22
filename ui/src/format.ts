export const score = (x: number | null | undefined) => (x == null ? "-" : x.toFixed(3))

export const signed = (x: number | null | undefined) => (x == null ? "-" : `${x > 0 ? "+" : ""}${x.toFixed(3)}`)

export const params = (x: number | null | undefined) => (x == null ? "-" : `${(x / 1e6).toFixed(1)}M`)

export const ms = (x: number | null | undefined) => (x == null ? "-" : `${x.toFixed(1)} ms`)

export const when = (t: number | null | undefined) => (t == null ? "-" : new Date(t * 1000).toLocaleTimeString())

const MINUTE = 60
const HOUR = 60 * MINUTE

export function seconds(s: number): string {
  if (Math.round(s) < MINUTE) return `${Math.round(s)} s`
  if (s < HOUR) return `${Math.round(s / MINUTE)} min`
  return `${Math.floor(s / HOUR)} h ${Math.round((s % HOUR) / MINUTE)} min`
}

export function took(a: number | null, b: number | null): string {
  return a == null ? "-" : seconds((b ?? Date.now() / 1000) - a)
}

const RIVAL_NAMES: Record<string, string> = { llm: "the LLM", jev: "Jev", panel: "the teacher panel", silero: "Silero", energy: "an energy threshold" }

export function rivalName(id: string | null | undefined): string {
  if (!id) return "-"
  if (RIVAL_NAMES[id]) return RIVAL_NAMES[id]
  const repo = id.startsWith("hf:") ? id.slice(3) : id
  return repo.slice(repo.lastIndexOf("/") + 1)
}

export const metricName = (metric: string) => metric.split("_").join(" ").replace("f1", "F1")
