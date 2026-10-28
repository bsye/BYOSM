import { useState } from "react"
import { post, upload } from "@/api"

export type PageRead = { label: string; probs: Record<string, number>; page: string; pages: number; text: string; grid: number }
type Row = { row: number; drops: number[] }
type Answer = { label: string; p: number }
export type Without = { picture: Answer; text: Answer }

export function usePageReading(name: string) {
  const [page, setPage] = useState<PageRead | null>(null)
  const [rows, setRows] = useState<Record<number, number[]>>({})
  const [without, setWithout] = useState<Without | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const scoreRows = (file: string, read: PageRead) =>
    Array.from({ length: read.grid }, (_, r) =>
      post<Row>(`/specialists/${name}/page/${r}`, { file, label: read.label }).then((x) => setRows((all) => ({ ...all, [x.row]: x.drops }))),
    )
  const send = (f: File) => {
    setBusy(true)
    setError("")
    setPage(null)
    setRows({})
    setWithout(null)
    upload<{ file: string }>(`/specialists/${name}/tried`, f)
      .then(({ file }) =>
        post<PageRead>(`/specialists/${name}/page`, { file }).then((read) => {
          setPage(read)
          post<Without>(`/specialists/${name}/page/views`, { file, label: read.label }).then(setWithout)
          scoreRows(file, read)
        }),
      )
      .catch((e) => setError(String(e)))
      .finally(() => setBusy(false))
  }
  return { page, rows, without, busy, error, send }
}
