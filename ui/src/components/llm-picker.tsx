import { useEffect, useState } from "react"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { choiceKey, DEFAULT_MODEL, roleConfig, useProviders, type Provider } from "@/providers"

const CHOICE_KEY = "byosm-llm-choice"
export const STANDARD = "standard"
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

type SelectProps = { providers: Provider[] | null; value: string; onChange: (v: string) => void; label: string; standard?: boolean; className?: string }

export function LlmSelect({ providers, value, onChange, label, standard, className }: SelectProps) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger size="sm" className={className ?? "max-w-72"} aria-label={label}>
        <SelectValue placeholder={providers == null ? "Looking for LLMs..." : "No LLM found"} />
      </SelectTrigger>
      <SelectContent>
        {standard && <SelectItem value={STANDARD}>Standard</SelectItem>}
        {(providers ?? []).map((p) => (
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

export function LlmPicker({ llm, label }: { llm: LlmChoice; label: string }) {
  return <LlmSelect providers={llm.providers} value={llm.choice} onChange={llm.pick} label={label} />
}

export function useStandardChoice(): { providers: Provider[] | null; value: string; pick: (v: string) => void; config: () => Record<string, unknown> | null } {
  const providers = useProviders()
  const [value, pick] = useState(STANDARD)
  const available = (providers ?? []).filter((p) => p.available)
  return { providers, value, pick, config: () => (value === STANDARD ? null : roleConfig(available, value)) }
}
