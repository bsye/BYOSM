import { Hammer, Lightbulb } from "lucide-react"
import { useState } from "react"
import { runJob } from "@/api"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { Button } from "@/components/ui/button"
import { InputGroup, InputGroupAddon, InputGroupTextarea } from "@/components/ui/input-group"
import { go } from "@/route"

const EXAMPLES = [
  "classify support emails into billing, shipping, technical or other",
  "find the dates, amounts and parties in a lease agreement",
  "detect when someone starts and stops speaking in microphone audio",
  "sort scanned documents into invoice, receipt, contract and other",
]

export function Create() {
  const llm = useLlmChoice()
  const [prompt, setPrompt] = useState("")
  const [error, setError] = useState("")
  const start = () =>
    runJob(["auto", prompt.trim()], llm.config())
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
          <LlmPicker llm={llm} label="LLM used while building" />
          <Button size="sm" disabled={!prompt.trim() || !llm.choice} onClick={start}>
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
