import { ChevronDown, ListChecks, Play, Sparkles } from "lucide-react"
import { useState } from "react"
import { runJob, type Detail, type Proposal } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { openJob } from "@/job-sheet"

export type NextStep = { id: string; title: string; why: string; args: string[]; experiments?: string[]; gain?: number | null; unblocks?: string[] }

const VALUE_CHARS = 60
const EXPERIMENTS = "experiments"

type Start = (args: string[]) => void

const shown = (v: unknown) => {
  const text = typeof v === "string" ? v : JSON.stringify(v)
  return text.length > VALUE_CHARS ? `${text.slice(0, VALUE_CHARS)}...` : text
}

function Experiment({ name, p, start }: { name: string; p: Proposal; start: Start }) {
  return (
    <li className="flex items-start gap-3 py-2.5">
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium">{p.id}</span>
          {p.family && <Badge variant="secondary">{p.family}</Badge>}
          {p.cost && <Badge variant="outline">about {Math.round(p.cost.minutes)} min</Badge>}
        </div>
        <p className="text-muted-foreground text-xs">{p.why}</p>
        <div className="flex flex-wrap gap-1">
          {p.ops.map((o, i) => (
            <Badge key={i} variant="secondary" className="font-mono font-normal">
              {o.path} = {shown(o.value)}
            </Badge>
          ))}
        </div>
      </div>
      <Button size="sm" variant="outline" onClick={() => start(["improve", name, "--apply", p.id, "--run"])}>
        <Play />
        Run
      </Button>
    </li>
  )
}

function StepRow({ i, s, name, proposals, start }: { i: number; s: NextStep; name: string; proposals: Proposal[]; start: Start }) {
  const [open, setOpen] = useState(false)
  const lined = proposals.filter((p) => s.experiments?.includes(p.id))
  return (
    <Collapsible open={open} onOpenChange={setOpen} asChild>
      <li className="py-2.5">
        <div className="flex items-start gap-3">
          <span className="text-muted-foreground w-4 shrink-0 pt-0.5 text-xs tabular-nums">{i + 1}</span>
          <div className="min-w-0 flex-1">
            <p className="font-medium">{s.title}</p>
            <p className="text-muted-foreground text-xs">{s.why}</p>
          </div>
          {lined.length > 0 && (
            <CollapsibleTrigger asChild>
              <Button size="sm" variant="ghost">
                <ChevronDown className={open ? "rotate-180" : ""} />
                {open ? "Hide" : "Show"}
              </Button>
            </CollapsibleTrigger>
          )}
          <Button size="sm" variant="outline" onClick={() => start(s.args)}>
            <Play />
            {s.id === EXPERIMENTS ? "Run all" : "Run"}
          </Button>
        </div>
        <CollapsibleContent>
          <ul className="mt-2 ml-7 divide-y border-t">
            {lined.map((p) => (
              <Experiment key={p.id} name={name} p={p} start={start} />
            ))}
          </ul>
        </CollapsibleContent>
      </li>
    </Collapsible>
  )
}

type Steps = { name: string; steps: NextStep[]; proposals: Proposal[]; ask: string[]; batch: boolean }

export function NextSteps({ d }: { d: Detail }) {
  return <StepsCard name={d.name} steps={d.next as NextStep[]} proposals={d.proposals?.proposals ?? []} ask={["improve", d.name]} batch />
}

export function StepsCard({ name, steps, proposals, ask, batch }: Steps) {
  const llm = useLlmChoice()
  const start: Start = (args) => void runJob(args, llm.config() ?? undefined).then((j) => openJob(j.id))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ListChecks className="size-4" />
          Next steps
        </CardTitle>
        <CardAction className="flex gap-2">
          <LlmPicker llm={llm} label="LLM for these steps" />
          <Button size="sm" variant="outline" onClick={() => start(ask)}>
            <Sparkles />
            Ask the LLM
          </Button>
          {batch && steps.length > 1 && (
            <Button size="sm" onClick={() => start(["next", name, "--all"])}>
              <Play />
              Do all {steps.length}
            </Button>
          )}
        </CardAction>
      </CardHeader>
      {steps.length > 0 && (
        <CardContent>
          <ol className="divide-y">
            {steps.map((s, i) => (
              <StepRow key={s.id} i={i} s={s} name={name} proposals={proposals} start={start} />
            ))}
          </ol>
        </CardContent>
      )}
    </Card>
  )
}
