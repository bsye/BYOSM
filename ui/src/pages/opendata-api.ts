import { useEffect, useState } from "react"
import { api, post, runJob } from "@/api"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"

export type OpenFile = { name: string; format: string; url: string }
export type OpenDataset = { source: string; id: string; title: string; publisher: string; licence: string; modified: string; url: string; notes?: string; files: OpenFile[] }
export type Kept = OpenDataset & { file: OpenFile; added: number; cached: { files: string[] } | null }
export type Column = { name: string; type: string; missing: number; distinct: number; examples: string[] }
export type Preview = { file: string; files: string[]; sampled: number; columns: Column[]; rows: Record<string, string | null>[] }

export const download = (url: string) => runJob(["opendata", "fetch", url]).then((j) => openJob(j.id))
export const keep = (dataset: OpenDataset, file: OpenFile) => post<Kept[]>("/opendata/library", { dataset, file })
export const forget = (url: string) => post<Kept[]>("/opendata/library/remove", { url })

export function useLibrary(): [Kept[] | null, (k: Kept[]) => void] {
  const fetched = Object.values(useLive().jobs).filter((j) => j.args[0] === "opendata" && j.status === "done").length
  const [kept, setKept] = useState<Kept[] | null>(null)
  useEffect(() => {
    api<Kept[]>("/opendata/library").then(setKept, () => setKept([]))
  }, [fetched])
  return [kept, setKept]
}
