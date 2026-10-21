export const score = (x: number | null | undefined) => (x == null ? "-" : x.toFixed(3))

export const params = (x: number | null | undefined) => (x == null ? "-" : `${(x / 1e6).toFixed(1)}M`)

export const ms = (x: number | null | undefined) => (x == null ? "-" : `${x.toFixed(1)} ms`)

export const when = (t: number | null | undefined) => (t == null ? "-" : new Date(t * 1000).toLocaleTimeString())

export const took = (a: number | null, b: number | null) =>
  a == null ? "-" : `${Math.round(((b ?? Date.now() / 1000) - a) / 6) / 10} min`
