import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useLive } from "@/events"
import { score } from "@/format"
import { go, href } from "@/route"
import { Overview } from "./Overview"

function NewSpecialist() {
  const [prompt, setPrompt] = useState("")
  const [error, setError] = useState("")
  const start = async () => {
    try {
      const job = await runJob(["auto", prompt])
      go("jobs", job.id)
    } catch (e) {
      setError(String(e))
    }
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>New specialist</CardTitle>
        <CardDescription>One sentence. The autopilot infers the task, finds a benchmark and rivals, and trains until it wins or the budget ends.</CardDescription>
      </CardHeader>
      <CardContent>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="classify support emails into billing, shipping, technical or other"
        />
        {error && <p className="text-destructive mt-2 text-sm">{error}</p>}
      </CardContent>
      <CardFooter>
        <Button disabled={!prompt.trim()} onClick={start}>
          Build it
        </Button>
      </CardFooter>
    </Card>
  )
}

export function Specialists() {
  const [items, setItems] = useState<Summary[] | null>(null)
  const finished = Object.values(useLive().jobs).filter((j) => j.status === "done").length
  useEffect(() => {
    api<Summary[]>("/specialists").then(setItems)
  }, [finished])
  return (
    <div className="space-y-4">
      <Overview items={items ?? []} />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      <NewSpecialist />
      {(items ?? []).map((s) => (
        <a key={s.name} href={href("specialists", s.name)}>
          <Card className="hover:bg-accent/50 h-full transition-colors">
            <CardHeader>
              <CardTitle className="flex items-center justify-between gap-2">
                {s.name}
                <span className="font-mono text-2xl">{score(s.score)}</span>
              </CardTitle>
              <CardDescription className="line-clamp-2">{s.description}</CardDescription>
            </CardHeader>
            <CardFooter className="flex flex-wrap gap-2">
              <Badge variant="secondary">{s.modality}</Badge>
              <Badge variant="secondary">{s.task_type}</Badge>
              <Badge variant="outline">
                {s.metric} on {s.primary_suite}
              </Badge>
              {s.exported && <Badge>exported</Badge>}
              {s.rival?.score != null && s.score != null && (
                <Badge variant="outline">
                  {s.score > s.rival.score ? "+" : ""}
                  {(s.score - s.rival.score).toFixed(3)} vs {s.rival.name}
                </Badge>
              )}
            </CardFooter>
          </Card>
        </a>
      ))}
      </div>
    </div>
  )
}
