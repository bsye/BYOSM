import { useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import type { Today } from "./ForecastToday"

export type Score = { auc: number; auc_ci: [number, number]; ap: number; brier: number; base_rate: number; predicted_rate?: number }
export type Leak = { feature: string; auc: number; suspect: boolean; why: string }
export type Spec = { title: string; shown?: string[]; horizon_years: number; cutoffs: Record<string, number[]>; excluded: Record<string, string> }
export type Year = { year: number; positive: number; scores: Record<string, Score> }
export type Result = {
  at: number
  today?: Today
  over_time?: Year[]
  dropped?: Record<string, string>
  splits: Record<string, { n: number; positive: number }>
  leakage: Leak[]
  scores: Record<string, Score>
}
export type Forecast = { name: string; spec: Spec; result: Result | null }

const NEVER_DAYS = 9999
export const featureValue = (v: string | number | null) =>
  v == null ? "missing" : v === NEVER_DAYS ? "never" : typeof v === "number" ? String(Math.round(v * 10) / 10) : v

export const runForecast = (name: string) => runJob(["forecast", "run", name]).then((j) => openJob(j.id))
export const within = (years: number) => (years === 1 ? "within a year" : `within ${years} years`)
export const best = (f: Forecast) => f.result?.scores["gradient boosting"]

export function useForecasts(): [Forecast[] | null, Set<string>, () => void] {
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "forecast")
  const running = new Set(jobs.filter((j) => j.status === "running" || j.status === "queued").map((j) => j.args[2]))
  const done = jobs.filter((j) => j.status === "done").length
  const [rows, setRows] = useState<Forecast[] | null>(null)
  const [saved, setSaved] = useState(0)
  useEffect(() => {
    api<Forecast[]>("/forecasts").then(setRows, () => setRows([]))
  }, [done, saved])
  return [rows, running, () => setSaved((n) => n + 1)]
}
