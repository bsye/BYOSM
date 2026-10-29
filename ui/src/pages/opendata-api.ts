import { useEffect, useState } from "react"
import { api, post, runJob } from "@/api"
import { useFinished, useLive } from "@/events"
import { openJob } from "@/job-sheet"

const POLL_MS = 3000

export type OpenFile = { name: string; format: string; url: string }
export type OpenDataset = {
  source: string
  id: string
  title: string
  publisher: string
  licence: string
  modified: string
  url: string
  notes?: string
  why?: string
  files: OpenFile[]
}
export type Kept = OpenDataset & { file: OpenFile; added: number; cached: { files: string[] } | null }
export type Column = { name: string; type: string; missing: number; distinct: number; examples: string[] }
export type Preview = { file?: string; files?: string[]; match?: number; sampled: number; columns: Column[]; rows: Record<string, string | null>[] }
export type Portal = { name: string; url: string; why?: string }
export type DataAsk = { request: string; at: number; status: "running" | "done" | "failed"; queries: string[]; datasets: OpenDataset[]; portals: Portal[] }
export type Source = { name: string; url: string; kind: "ckan" | "socrata" | "page"; by: string; why?: string }
export type KeyPair = { left: string; right: string; match: number }

export const download = (url: string) => runJob(["opendata", "fetch", url]).then((j) => openJob(j.id))
export const keep = (dataset: OpenDataset, file: OpenFile) => post<Kept[]>("/opendata/library", { dataset, file })
export const forget = (url: string) => post<Kept[]>("/opendata/library/remove", { url })

export function useLibrary(): [Kept[] | null, (k: Kept[]) => void] {
  const fetched = useFinished("opendata")
  const [kept, setKept] = useState<Kept[] | null>(null)
  useEffect(() => {
    api<Kept[]>("/opendata/library").then(setKept, () => setKept([]))
  }, [fetched])
  return [kept, setKept]
}

export const ask = (request: string) => runJob(["opendata", "find", request])
export const forgetAsk = (request: string) => post<DataAsk[]>("/opendata/requests/remove", { request })
export const addSource = (url: string, name: string) => post<Source[]>("/opendata/sources", { url, name })
export const dropSource = (url: string) => post<Source[]>("/opendata/sources/remove", { url })

export function useAsks(): [DataAsk[] | null, (a: DataAsk[]) => void] {
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "opendata" && j.args[1] === "find")
  const running = jobs.some((j) => j.status === "running" || j.status === "queued")
  const [asks, setAsks] = useState<DataAsk[] | null>(null)
  const ended = useFinished("opendata")
  useEffect(() => {
    const load = () => api<DataAsk[]>("/opendata/requests").then(setAsks, () => setAsks([]))
    load()
    if (!running) return
    const timer = setInterval(load, POLL_MS)
    return () => clearInterval(timer)
  }, [ended, running])
  return [asks, setAsks]
}
