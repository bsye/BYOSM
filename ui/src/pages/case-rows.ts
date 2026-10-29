import type { Detail } from "@/api"
import { suiteSeries } from "./suite-rows"

export type CaseRow = { case: string } & Record<string, number | [number, number] | string | undefined>
export type CaseModel = { model: string; index: number }

export const caseSuites = (d: Detail) => d.suites.map((s) => s.name).filter((s) => Object.values(d.board).some((by) => by[s]?.cases))

export function caseSeries(d: Detail, suite: string): { models: CaseModel[]; rows: CaseRow[] } {
  const models = suiteSeries(d).models.flatMap((model, index) => (d.board[model]?.[suite]?.cases ? [{ model, index }] : []))
  const cases = [...new Set(models.flatMap(({ model }) => Object.keys(d.board[model][suite].cases ?? {})))].sort()
  const rows = cases.map((c) => {
    const row: CaseRow = { case: c.replaceAll("_", " ") }
    for (const { model, index } of models) {
      const v = d.board[model][suite].cases?.[c]
      if (v?.accuracy == null) continue
      row[`s${index}`] = v.accuracy
      row[`s${index}ci`] = [v.accuracy - v.ci[0], v.ci[1] - v.accuracy]
    }
    return row
  })
  return { models, rows }
}
