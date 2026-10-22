import { useEffect, useState } from "react"
import { api } from "@/api"

export type Provider = {
  id: string
  label: string
  kind: string
  available: boolean
  models: string[]
  config: Record<string, unknown> | null
  hint?: string
}

export function useProviders(): Provider[] | null {
  const [list, setList] = useState<Provider[] | null>(null)
  useEffect(() => {
    api<{ providers: Provider[] }>("/llm/providers")
      .then((r) => setList(r.providers))
      .catch(() => setList([]))
  }, [])
  return list
}

export const DEFAULT_MODEL = "default"

export const choiceKey = (p: Provider, model: string | null) => `${p.id}::${model ?? DEFAULT_MODEL}`

export function roleConfig(providers: Provider[], key: string): Record<string, unknown> | null {
  const [id, model] = key.split("::")
  const p = providers.find((x) => x.id === id && x.available)
  if (!p?.config) return null
  return model === DEFAULT_MODEL ? p.config : { ...p.config, model }
}

export function matchChoice(providers: Provider[], conf: Record<string, unknown>): string {
  const p = providers.find((x) => x.config && x.config.provider === conf.provider && (x.config.base_url ?? null) === (conf.base_url ?? null))
  return p ? choiceKey(p, (conf.model as string | null) ?? null) : ""
}
