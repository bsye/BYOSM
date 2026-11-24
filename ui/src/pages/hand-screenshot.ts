import { useCallback, useState } from "react"
import { post } from "@/api"
import { drafted, type Draft, type Seen } from "./hand-builder"

const IMAGE = "image/"

function dataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error("the picture could not be opened"))
    reader.readAsDataURL(file)
  })
}

export const pastedImage = (items: DataTransferItemList) =>
  Array.from(items)
    .find((item) => item.type.startsWith(IMAGE))
    ?.getAsFile() ?? null

export function useScreenshot(blinds: [string, string], onRead: (draft: Draft) => void) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const read = useCallback(
    (file: File) => {
      setBusy(true)
      setError("")
      dataUrl(file)
        .then((image) => post<Seen>("/poker/screenshot", { image }))
        .then(
          (seen) => onRead(drafted(seen, blinds)),
          (e) => setError(e instanceof Error ? e.message : String(e)),
        )
        .finally(() => setBusy(false))
    },
    [blinds, onRead],
  )
  return { busy, error, read }
}
