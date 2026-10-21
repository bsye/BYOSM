export const score = (x: number | null | undefined) => (x == null ? "-" : x.toFixed(3))

export const params = (x: number | null | undefined) => (x == null ? "-" : `${(x / 1e6).toFixed(1)}M`)

export const ms = (x: number | null | undefined) => (x == null ? "-" : `${x.toFixed(1)} ms`)

export const when = (t: number | null | undefined) => (t == null ? "-" : new Date(t * 1000).toLocaleTimeString())

const MINUTE = 60

export function took(a: number | null, b: number | null): string {
  if (a == null) return "-"
  const s = (b ?? Date.now() / 1000) - a
  return s < MINUTE ? `${Math.round(s)} s` : `${Math.round((s / MINUTE) * 10) / 10} min`
}
