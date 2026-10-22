import { ListChecks, Play } from "lucide-react"
import { runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { openJob } from "@/job-sheet"

export type NextStep = { id: string; title: string; why: string; args: string[] }

const start = (args: string[]) => runJob(args).then((j) => openJob(j.id))

export function NextSteps({ name, steps }: { name: string; steps: NextStep[] }) {
  if (!steps.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ListChecks className="size-4" />
          Next steps
        </CardTitle>
        {steps.length > 1 && (
          <CardAction>
            <Button size="sm" onClick={() => start(["next", name, "--all"])}>
              <Play />
              Do all {steps.length}
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent>
        <ol className="divide-y">
          {steps.map((s, i) => (
            <li key={s.id} className="flex items-start gap-3 py-2.5">
              <span className="text-muted-foreground w-4 shrink-0 pt-0.5 text-xs tabular-nums">{i + 1}</span>
              <div className="min-w-0 flex-1">
                <p className="font-medium">{s.title}</p>
                <p className="text-muted-foreground text-xs">{s.why}</p>
              </div>
              <Button size="sm" variant="outline" onClick={() => start(s.args)}>
                <Play />
                Run
              </Button>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}
