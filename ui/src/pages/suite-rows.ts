import type { Detail } from "@/api"

export const MAX_SERIES = 4
const MIN_SUITES = 2

export type SuiteRow = { suite: string } & Record<string, number | [number, number] | string | undefined>

const scoredOn = (d: Detail, model: string) => d.suites.filter((s) => d.board[model]?.[s.name]?.objective != null).length

export function suiteSeries(d: Detail): { models: string[]; rows: SuiteRow[] } {
  const primaryTest = (m: string) => d.board[m]?.[d.primary_suite]?.objective ?? 0
  const rivals = d.baselines.filter((m) => scoredOn(d, m) >= MIN_SUITES)
  const others = Object.keys(d.board)
    .filter((m) => m !== d.promoted && !d.baselines.includes(m) && scoredOn(d, m) >= MIN_SUITES)
    .sort((a, b) => primaryTest(b) - primaryTest(a))
  const promoted = d.promoted && scoredOn(d, d.promoted) >= MIN_SUITES ? [d.promoted] : []
  const models = [...promoted, ...rivals, ...others].slice(0, MAX_SERIES)
  const rows = d.suites.map((s) => {
    const row: SuiteRow = { suite: s.name }
    models.forEach((m, i) => {
      const r = d.board[m]?.[s.name]
      if (r?.objective == null) return
      row[`s${i}`] = r.objective
      row[`s${i}ci`] = r.ci ? [r.objective - r.ci.lo, r.ci.hi - r.objective] : [0, 0]
    })
    return row
  })
  return { models: models.length >= 1 && d.suites.length >= MIN_SUITES ? models : [], rows }
}
