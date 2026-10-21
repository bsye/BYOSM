import { useEffect, useState } from "react"
import { readToken } from "@/api"

function parts(): string[] {
  readToken()
  const hash = window.location.hash.slice(1)
  const path = hash.startsWith("/") ? hash.slice(1) : hash
  return path.split("/").filter(Boolean).map(decodeURIComponent)
}

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
