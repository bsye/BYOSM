export const score = (x: number | null | undefined) => (x == null ? "-" : x.toFixed(3))

export const signed = (x: number | null | undefined) => (x == null ? "-" : `${x > 0 ? "+" : ""}${x.toFixed(3)}`)

export const params = (x: number | null | undefined) => (x == null ? "-" : `${(x / 1e6).toFixed(1)}M`)

export const ms = (x: number | null | undefined) => (x == null ? "-" : `${x.toFixed(1)} ms`)

export const when = (t: number | null | undefined) => (t == null ? "-" : new Date(t * 1000).toLocaleTimeString())

const MINUTE = 60

export function seconds(s: number): string {
  if (Math.round(s) < MINUTE) return `${Math.round(s)} s`
  const minutes = Math.round(s / MINUTE)
  if (minutes < MINUTE) return `${minutes} min`
  return `${Math.floor(minutes / MINUTE)} h ${minutes % MINUTE} min`
}

export function took(a: number | null, b: number | null): string {
  return a == null ? "-" : seconds((b ?? Date.now() / 1000) - a)
}

const RIVAL_NAMES: Record<string, string> = { llm: "the LLM", jev: "Jev", panel: "the teacher panel", silero: "Silero", energy: "an energy threshold" }

export function rivalName(id: string | null | undefined): string {
  if (!id) return "-"
  if (RIVAL_NAMES[id]) return RIVAL_NAMES[id]
  const repo = (id.startsWith("hf:") ? id.slice(3) : id).split("#")[0]
  return repo.slice(repo.lastIndexOf("/") + 1)
}

const THOUSAND = 1000

export function tokens(n: number | null | undefined): string {
  if (n == null) return "-"
  if (n < THOUSAND) return String(n)
  if (n < THOUSAND * THOUSAND) return `${(n / THOUSAND).toFixed(1)}k`
  return `${(n / THOUSAND / THOUSAND).toFixed(2)}M`
}

export const gigabytes = (bytes: number | null | undefined) => (bytes == null ? "-" : `${(bytes / 1e9).toFixed(1)} GB`)
