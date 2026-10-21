import { useEffect, useState } from "react"

const parts = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent)

export function useRoute(): string[] {
  const [route, setRoute] = useState(parts)
  useEffect(() => {
    const on = () => setRoute(parts())
    window.addEventListener("hashchange", on)
    return () => window.removeEventListener("hashchange", on)
  }, [])
  return route
}

export const go = (...path: string[]) => {
  window.location.hash = "/" + path.map(encodeURIComponent).join("/")
}

export const href = (...path: string[]) => "#/" + path.map(encodeURIComponent).join("/")
