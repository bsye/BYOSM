import { ArrowRight } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { href } from "@/route"

type Node = { specialist: string; role: string; task: string }
type Tool = { when: string; args: Record<string, string>; call: string | null }

const ORDER = ["router", "tagger"]

function Step({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="min-w-48 flex-1">
      <CardHeader className="pb-2">
        <CardDescription>{title}</CardDescription>
        {children}
      </CardHeader>
    </Card>
  )
}

export function SystemDiagram({ nodes, tools }: { nodes: Record<string, Node>; tools: Record<string, Tool> }) {
  const steps = Object.values(nodes).sort((a, b) => ORDER.indexOf(a.role) - ORDER.indexOf(b.role))
  return (
    <div className="flex flex-col items-stretch gap-2 lg:flex-row">
      <Step title="Input">
        <CardTitle className="text-base">Text, as typed</CardTitle>
      </Step>
      {steps.map((n) => (
        <div key={n.specialist} className="contents">
          <ArrowRight className="text-muted-foreground mx-auto shrink-0 rotate-90 self-center lg:rotate-0" />
          <Step title={n.role === "router" ? "Picks the action" : "Finds the arguments"}>
            <CardTitle className="text-base">
              <a className="hover:underline" href={href("specialists", n.specialist)}>
                {n.specialist}
              </a>
            </CardTitle>
          </Step>
        </div>
      ))}
      <ArrowRight className="text-muted-foreground mx-auto shrink-0 rotate-90 self-center lg:rotate-0" />
      <Card className="min-w-56 flex-1">
        <CardHeader className="pb-2">
          <CardDescription>Your code</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {Object.entries(tools).map(([name, t]) => (
            <div key={name} className="flex flex-wrap items-center gap-1 text-xs">
              <Badge variant="outline">{t.when}</Badge>
              <ArrowRight className="size-3" />
              <span className="font-mono">
                {name}({Object.keys(t.args).join(", ")})
              </span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
