import { useEffect, useState } from "react"
import { api, type Summary } from "@/api"
import { useLive } from "@/events"
import { seconds } from "@/format"
import type { HealthReport } from "./Health"
import type { NextStep } from "./NextSteps"

export type Score = {
  mase: number
  crps: number
  calib_mase?: number
  train_s?: number
  ms_per_forecast?: number
  published?: boolean
  datasets?: number
  ours_on_shared?: number
}
type Predict = { rates?: number[]; routing?: boolean }
type Fit = { pretraining?: string; quantile_boosting?: boolean; lab_method?: string | null }
export type Version = {
  name: string
  recipe: string
  effort?: number
  predict?: Predict
  fit?: Fit
  calib: number
  test: number
  crps: number
  ci: [number, number] | null
  at: number
}
export type ForecasterView = Summary & {
  configs?: Record<string, Record<string, Score>>
  overall?: Record<string, Score>
  pending?: string[]
  versions?: Version[]
  health?: HealthReport | null
  next?: NextStep[]
  live?: string | null
  routes?: Record<string, string>
}

const RIVALS = ["seasonal naive", "Timestral"]
export const isRival = (model: string) => RIVALS.some((r) => model.startsWith(r))
export const trainTime = (s?: number) => (s == null ? "-" : seconds(s))

export function useForecaster(name: string): ForecasterView | null | undefined {
  const [view, setView] = useState<ForecasterView | null | undefined>(undefined)
  const live = useLive()
  const progress = Object.values(live.jobs).reduce(
    (n, j) => n + (live.metrics[j.id] ?? []).filter((m) => m.kind === "series").length + (j.status === "done" ? 1 : 0),
    0,
  )
  useEffect(() => {
    api<ForecasterView>(`/series/${name}`)
      .then(setView)
      .catch(() => setView(null))
  }, [name, progress])
  return view
}
