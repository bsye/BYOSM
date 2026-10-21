import { Hammer, Lightbulb } from "lucide-react"
import { useEffect, useState } from "react"
import { runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { InputGroup, InputGroupAddon, InputGroupTextarea } from "@/components/ui/input-group"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { choiceKey, DEFAULT_MODEL, roleConfig, useProviders } from "@/providers"
import { go } from "@/route"

const EXAMPLES = [
  "classify support emails into billing, shipping, technical or other",
  "find the dates, amounts and parties in a lease agreement",
  "detect when someone starts and stops speaking in microphone audio",
  "sort scanned documents into invoice, receipt, contract and other",
]
const CHOICE_KEY = "byosm-llm-choice"
const PREFERRED = "ollama"

function remembered(): string {
  try {
    return localStorage.getItem(CHOICE_KEY) ?? ""
  } catch {
    return ""
  }
}

export function Create() {
  const providers = useProviders()
  const [prompt, setPrompt] = useState("")
  const [choice, setChoice] = useState(remembered)
  const [error, setError] = useState("")
  const available = (providers ?? []).filter((p) => p.available)
  useEffect(() => {
    const first = available.find((p) => p.id === PREFERRED) ?? available[0]
    const known = available.some((p) => [...p.models, null].some((m) => choiceKey(p, m) === choice))
    if (first && !known) setChoice(choiceKey(first, first.models[0] ?? null))
  }, [providers])
  const pick = (v: string) => {
    setChoice(v)
    try {
      localStorage.setItem(CHOICE_KEY, v)
    } catch {
      return
    }
  }
  const start = () =>
    runJob(["auto", prompt.trim()], roleConfig(available, choice))
      .then((j) => go("jobs", j.id))
      .catch((e) => setError(String(e)))
  return (
    <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center gap-6 py-12">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-semibold">What should your specialist do?</h1>
        <p className="text-muted-foreground">Describe the task in one sentence.</p>
      </div>
      <InputGroup>
        <InputGroupTextarea
          className="min-h-24"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && prompt.trim()) start()
          }}
          placeholder={EXAMPLES[0]}
        />
        <InputGroupAddon align="block-end" className="justify-between">
          <Select value={choice} onValueChange={pick}>
            <SelectTrigger size="sm" className="max-w-72" aria-label="LLM used while building">
              <SelectValue placeholder={providers == null ? "Looking for LLMs..." : "No LLM found"} />
            </SelectTrigger>
            <SelectContent>
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
          <Button size="sm" disabled={!prompt.trim() || !choice} onClick={start}>
            <Hammer />
            Build it
          </Button>
        </InputGroupAddon>
      </InputGroup>
      {error && <p className="text-destructive text-sm">{error}</p>}
      <div className="flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((x) => (
          <Button key={x} variant="outline" size="sm" className="text-muted-foreground" onClick={() => setPrompt(x)}>
            <Lightbulb />
            {x}
          </Button>
        ))}
      </div>
    </div>
  )
}
