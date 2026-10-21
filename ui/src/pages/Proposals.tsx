import { runJob, type Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { go } from "@/route"

const VALUE_CHARS = 60

const shown = (v: unknown) => {
  const text = typeof v === "string" ? v : JSON.stringify(v)
  return text.length > VALUE_CHARS ? `${text.slice(0, VALUE_CHARS)}...` : text
}

export function Proposals({ d }: { d: Detail }) {
  const saved = d.proposals
  const ask = () => runJob(["improve", d.name]).then((j) => go("jobs", j.id))
  const run = (id: string) => runJob(["improve", d.name, "--apply", id, "--run"]).then((j) => go("jobs", j.id))
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="space-y-1.5">
          <CardTitle>What to try next</CardTitle>
          <CardDescription>
            {saved
              ? `Diagnosed from ${saved.from}${saved.from !== d.promoted ? `, not the promoted ${d.promoted}: ask again for fresh ones` : ""}. Each is one spec edit with its evidence.`
              : "No proposals yet. The diagnosis reads the board, the training curve and the confusions."}
          </CardDescription>
        </div>
        <Button variant="outline" onClick={ask}>
          Ask for proposals
        </Button>
      </CardHeader>
      {saved && saved.proposals.length > 0 && (
        <CardContent className="space-y-4">
          {saved.proposals.map((p) => (
            <div key={p.id} className="flex flex-col gap-2 rounded-lg border p-4 md:flex-row md:items-start md:justify-between">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{p.id}</span>
                  {p.cost && <Badge variant="outline">about {Math.round(p.cost.minutes)} min</Badge>}
                </div>
                <p className="text-muted-foreground text-sm">{p.why}</p>
                <div className="flex flex-wrap gap-1">
                  {p.ops.map((o, i) => (
                    <Badge key={i} variant="secondary" className="font-mono">
                      {o.path} = {shown(o.value)}
                    </Badge>
                  ))}
                </div>
              </div>
              <Button size="sm" onClick={() => run(p.id)}>
                Run it
              </Button>
            </div>
          ))}
        </CardContent>
      )}
    </Card>
  )
}
