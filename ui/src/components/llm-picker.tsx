import { useEffect, useState } from "react"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { choiceKey, DEFAULT_MODEL, roleConfig, useProviders, type Provider } from "@/providers"

const CHOICE_KEY = "byosm-llm-choice"
const PREFERRED = "ollama"

function remembered(): string {
  try {
    return localStorage.getItem(CHOICE_KEY) ?? ""
  } catch {
    return ""
  }
}

export type LlmChoice = {
  providers: Provider[] | null
  choice: string
  pick: (v: string) => void
  config: () => Record<string, unknown> | null
}

export function useLlmChoice(): LlmChoice {
  const providers = useProviders()
  const [choice, setChoice] = useState(remembered)
  const available = (providers ?? []).filter((p) => p.available)
  useEffect(() => {
    const first = available.find((p) => p.id === PREFERRED) ?? available[0]
    const known = available.some((p) => [...p.models, null].some((m) => choiceKey(p, m) === choice))
    if (first && !known) setChoice(choiceKey(first, first.models[0] ?? null))
  }, [providers, available, choice])
  const pick = (v: string) => {
    setChoice(v)
    try {
      localStorage.setItem(CHOICE_KEY, v)
    } catch {
      return
    }
  }
  return { providers, choice, pick, config: () => roleConfig(available, choice) }
}

export function LlmPicker({ llm, label }: { llm: LlmChoice; label: string }) {
  return (
    <Select value={llm.choice} onValueChange={llm.pick}>
      <SelectTrigger size="sm" className="max-w-72" aria-label={label}>
        <SelectValue placeholder={llm.providers == null ? "Looking for LLMs..." : "No LLM found"} />
      </SelectTrigger>
      <SelectContent>
        {(llm.providers ?? []).map((p) => (
          <SelectGroup key={p.id}>
            <SelectLabel>
              {p.label} <span className="text-muted-foreground/70">{p.available ? p.kind : p.hint ?? "not found"}</span>
            </SelectLabel>
            {(p.models.length ? p.models : [null]).map((m) => (
              <SelectItem key={m ?? DEFAULT_MODEL} value={choiceKey(p, m)} disabled={!p.available}>
                {m ?? `${p.label}, its default model`}
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  )
}
