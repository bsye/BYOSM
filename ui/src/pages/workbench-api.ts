import { useCallback, useEffect, useState } from "react"
import { api, put } from "@/api"
import { useLive } from "@/events"

export type Row = { id: string; input: string; label: string; source?: string }
export type Answer = { label: string; confidence: number }
export type Score = { right: number; scored: number }
export type TestSet = { suite: string; frozen: boolean; board: boolean; total: number; scores: Record<string, Score> }
export type Results = Record<string, Record<string, Answer>>
export type RowsPage = { total: number; rows: Row[]; results: Results }
export type ModelChoice = { chosen: string[]; available: string[] }

const UNFINISHED = new Set(["running", "queued"])

export function useWorkbench(name: string) {
  const live = useLive()
  const jobs = Object.values(live.jobs).filter((j) => (j.args[0] === "inputs" || j.args[0] === "testset") && j.args.includes(name))
  const running = jobs.filter((j) => UNFINISHED.has(j.status))
  const ended = jobs.length - running.length
  const scored = running.reduce((n, j) => n + (live.metrics[j.id] ?? []).filter((m) => m.kind === "scored" || m.kind === "writing").length, 0)
  const [sets, setSets] = useState<TestSet[]>([])
  const [version, setVersion] = useState(0)
  const [models, setModels] = useState<ModelChoice | null>(null)
  const load = useCallback(() => {
    api<TestSet[]>(`/specialists/${name}/written`)
      .then((s) => {
        setSets(s)
        setVersion((v) => v + 1)
      })
      .catch(() => setSets([]))
  }, [name])
  useEffect(() => {
    load()
  }, [load, ended, scored])
  const loadModels = useCallback(() => {
    api<ModelChoice>(`/specialists/${name}/test-models`)
      .then(setModels)
      .catch(() => setModels(null))
  }, [name])
  useEffect(() => {
    loadModels()
  }, [loadModels])
  const choose = (chosen: string[]) => put<ModelChoice>(`/specialists/${name}/test-models`, { models: chosen }).then(setModels)
  return { sets, models, choose, loadModels, version, running: running.length }
}

export const score = (set: TestSet, model: string): Score => set.scores[model] ?? { right: 0, scored: 0 }

export function rowsPage(name: string, suite: string) {
  return (offset: number, limit: number) =>
    api<RowsPage>(`/specialists/${name}/written/${encodeURIComponent(suite)}/rows?offset=${offset}&limit=${limit}`).then((p) => ({ total: p.total, rows: p.rows, extra: p.results }))
}

export function mergeResults(a: Results, b: Results): Results {
  const out: Results = { ...a }
  for (const [m, got] of Object.entries(b)) out[m] = { ...(out[m] ?? {}), ...got }
  return out
}
