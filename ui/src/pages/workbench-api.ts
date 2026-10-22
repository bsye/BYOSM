import { useCallback, useEffect, useState } from "react"
import { api, put } from "@/api"
import { useLive } from "@/events"

export type Row = { id: string; input: string; label: string; source?: string }
export type Answer = { label: string; confidence: number }
export type TestSet = { suite: string; frozen: boolean; board: boolean; rows: Row[]; results: Record<string, Record<string, Answer>> }
export type ModelChoice = { chosen: string[]; available: string[] }

const UNFINISHED = new Set(["running", "queued"])

export function useWorkbench(name: string) {
  const live = useLive()
  const jobs = Object.values(live.jobs).filter((j) => (j.args[0] === "inputs" || j.args[0] === "testset") && j.args.includes(name))
  const running = jobs.filter((j) => UNFINISHED.has(j.status))
  const ended = jobs.length - running.length
  const scored = running.reduce((n, j) => n + (live.metrics[j.id] ?? []).filter((m) => m.kind === "scored" || m.kind === "writing").length, 0)
  const [sets, setSets] = useState<TestSet[]>([])
  const [models, setModels] = useState<ModelChoice | null>(null)
  const load = useCallback(() => {
    api<TestSet[]>(`/specialists/${name}/written`)
      .then(setSets)
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
  return { sets, models, choose, loadModels, running: running.length }
}

export function accuracy(set: TestSet, model: string): { right: number; scored: number } {
  const got = set.results[model] ?? {}
  const scored = set.rows.filter((r) => got[r.id])
  return { right: scored.filter((r) => got[r.id].label === r.label).length, scored: scored.length }
}
