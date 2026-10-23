import { MemoryStick } from "lucide-react"
import { useEffect, useState } from "react"
import { api } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { gigabytes } from "@/format"

const REFRESH_MS = 15000
const GB = 1024 ** 3

type Memory = { available: number; wired: number; total: number; low_gb: number; training_gb: number }

export function MemoryBanner() {
  const [m, setM] = useState<Memory | null>(null)
  useEffect(() => {
    const load = () => void api<Memory>("/machine/memory").then(setM, () => undefined)
    load()
    const id = setInterval(load, REFRESH_MS)
    return () => clearInterval(id)
  }, [])
  if (!m || m.available / GB >= m.low_gb) return null
  return (
    <Alert variant="destructive">
      <MemoryStick />
      <AlertTitle>
        Memory is low: {gigabytes(m.available)} free of {gigabytes(m.total)}, the GPU holds {gigabytes(m.wired)}
      </AlertTitle>
      <AlertDescription>New trainings wait until {m.training_gb} GB is free. Stopping a job or a container frees memory.</AlertDescription>
    </Alert>
  )
}
