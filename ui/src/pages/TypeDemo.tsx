import { useEffect, useState } from "react"
import { Area, AreaChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts"
import { post } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Input } from "@/components/ui/input"

const DEBOUNCE_MS = 60

type Step = { chars: number; action: string | null; confidence: number }
type Typed = {
  trace: Step[]
  fired_at: number | null
  frame: { action: string | null; args: { type: string; text: string }[]; tool: string | null; call: Record<string, string | null> } | null
}

const config = { confidence: { label: "confidence", color: "var(--chart-1)" } } satisfies ChartConfig

export function TypeDemo({ system }: { system: string }) {
  const [text, setText] = useState("")
  const [out, setOut] = useState<Typed | null>(null)
  const [error, setError] = useState("")
  useEffect(() => {
    const t = setTimeout(() => {
      if (!text.trim()) return setOut(null)
      post<Typed>(`/systems/${system}/type`, { text })
        .then((r) => {
          setOut(r)
          setError("")
        })
        .catch((e) => setError(String(e)))
    }, DEBOUNCE_MS)
    return () => clearTimeout(t)
  }, [text, system])
  const fired = out?.fired_at ?? null
  const call = out?.frame?.tool
    ? `${out.frame.tool}(${Object.entries(out.frame.call)
        .map(([k, v]) => `${k}=${v == null ? "None" : JSON.stringify(v)}`)
        .join(", ")})`
    : null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Type to it</CardTitle>
        <CardDescription>
          The router runs on every keystroke; the gate fires the action once it is sure for long enough. Runs the exported
          bundle on this machine.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="wake me up at five am tomorrow" />
        {error && <p className="text-destructive text-sm">{error}</p>}
        {out && (
          <>
            <p className="font-mono text-sm">
              <span className={fired ? "bg-primary text-primary-foreground rounded-sm" : ""}>{text.slice(0, fired ?? 0)}</span>
              <span className="text-muted-foreground">{text.slice(fired ?? 0)}</span>
            </p>
            <div className="flex flex-wrap items-center gap-2">
              {fired ? (
                <Badge>
                  fired {out.frame?.action} after {fired} of {text.length} characters
                </Badge>
              ) : (
                <Badge variant="outline">not sure yet: {out.trace.at(-1)?.action}</Badge>
              )}
              {out.frame?.args.map((a, i) => (
                <Badge key={i} variant="secondary">
                  {a.type}: {a.text}
                </Badge>
              ))}
            </div>
            {call && <p className="font-mono text-sm">{call}</p>}
            <ChartContainer config={config} className="h-40 w-full">
              <AreaChart data={out.trace} margin={{ left: 8, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="chars" tickLine={false} axisLine={false} />
                <YAxis domain={[0, 1]} tickLine={false} axisLine={false} width={32} />
                <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => p?.[0]?.payload?.action ?? ""} />} />
                {fired && <ReferenceLine x={fired} stroke="var(--foreground)" strokeDasharray="4 4" />}
                <Area dataKey="confidence" stroke="var(--color-confidence)" fill="var(--color-confidence)" fillOpacity={0.2} isAnimationActive={false} />
              </AreaChart>
            </ChartContainer>
          </>
        )}
      </CardContent>
    </Card>
  )
}
