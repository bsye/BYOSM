import { useSyncExternalStore } from "react"

let current: string | null = null
const listeners = new Set<() => void>()

function set(id: string | null) {
  current = id
  for (const l of listeners) l()
}

export const openJob = (id: string) => set(id)
export const closeJob = () => set(null)

export function useOpenJob(): string | null {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    () => current,
  )
}
