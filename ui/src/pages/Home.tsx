import { ArrowRight, Hammer, Lightbulb } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { Button } from "@/components/ui/button"
import { InputGroup, InputGroupAddon, InputGroupTextarea } from "@/components/ui/input-group"
import { Skeleton } from "@/components/ui/skeleton"
import { useLive } from "@/events"
import { openJob } from "@/job-sheet"
import { href } from "@/route"
import { Machine } from "./Machine"
import { HealthList, useHealths } from "./HealthList"
import { Opportunities } from "./Opportunities"
import { Overview } from "./Overview"
import { NextActions } from "./NextActions"
import { SectionCards } from "./SectionCards"
import { HomeForecasts } from "./HomeForecasts"
import { Section } from "./Section"

const EXAMPLES = [
  "classify support emails into billing, shipping, technical or other",
  "find the dates, amounts and parties in a lease agreement",
  "detect when someone starts and stops speaking in microphone audio",
  "will an Irish company close within 3 years?",
]

const STAT_CARDS = 4

function useItems(): Summary[] | null {
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  const [items, setItems] = useState<Summary[] | null>(null)
  useEffect(() => {
    api<Summary[]>("/specialists").then((s) => setItems(s.filter((x) => !x.part_of)), () => setItems([]))
  }, [finished])
  return items
}

const PROMPT_ID = "home-prompt"

function Prompt({ prompt, setPrompt }: { prompt: string; setPrompt: (p: string) => void }) {
  const llm = useLlmChoice()
  const [error, setError] = useState("")
  const start = () =>
    runJob(["auto", prompt.trim()], llm.config())
      .then((j) => openJob(j.id))
      .catch((e) => setError(String(e)))
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 pt-6 pb-2 md:pt-10">
      <h1 className="text-center text-3xl font-semibold tracking-tight">What should we build?</h1>
      <InputGroup>
        <InputGroupTextarea
          id={PROMPT_ID}
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
      <div className="grid gap-2 @xl/main:grid-cols-2">
        {EXAMPLES.map((x) => (
          <Button key={x} variant="ghost" size="sm" className="text-muted-foreground justify-start" onClick={() => setPrompt(x)}>
            <Lightbulb />
            <span className="truncate">{x}</span>
          </Button>
        ))}
      </div>
    </div>
  )
}

function StatSkeletons() {
  return (
    <div className="grid auto-rows-fr grid-cols-2 gap-4 @5xl/main:grid-cols-4">
      {Array.from({ length: STAT_CARDS }, (_, i) => (
        <Skeleton key={i} className="h-36 rounded-xl" />
      ))}
    </div>
  )
}

function Models({ items, onPick }: { items: Summary[] | null; onPick: (p: string) => void }) {
  const healths = useHealths(items)
  const all = (
    <Button variant="ghost" size="sm" asChild>
      <a href={href("specialists")}>
        All specialists
        <ArrowRight />
      </a>
    </Button>
  )
  if (items?.length === 0) return null
  return (
    <Section title="Your models" action={all}>
      {items ? <SectionCards items={items} healths={healths} /> : <StatSkeletons />}
      <div className="grid gap-4 @5xl/main:grid-cols-2">
        {items ? <Overview items={items} /> : <Skeleton className="h-96 rounded-xl" />}
        {items ? <HealthList items={items} healths={healths} /> : <Skeleton className="h-96 rounded-xl" />}
      </div>
      <div className="grid gap-4 @5xl/main:grid-cols-2">
        {items ? <NextActions items={items} healths={healths} /> : <Skeleton className="h-96 rounded-xl" />}
        <Opportunities onPick={onPick} />
      </div>
    </Section>
  )
}

export function Home() {
  const items = useItems()
  const [prompt, setPrompt] = useState("")
  const pick = (p: string) => {
    setPrompt(p)
    const box = document.getElementById(PROMPT_ID)
    box?.scrollIntoView({ behavior: "smooth", block: "center" })
    box?.focus()
  }
  return (
    <div className="flex flex-col gap-8 pb-8">
      <Prompt prompt={prompt} setPrompt={setPrompt} />
      <Models items={items} onPick={pick} />
      <HomeForecasts />
      <Machine />
    </div>
  )
}
