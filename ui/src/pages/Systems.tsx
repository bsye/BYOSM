import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { api } from "@/api"
import { TypeDemo } from "./TypeDemo"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"

type Split = { exact_frame: number; action_accuracy: number; arg_f1: number }
type Stream = { gate: { min_confidence: number; stable_for: number } | null; test: { typed_share: number; early_share: number; wrong_per_100: number }; test_at_end: { wrong_per_100: number }; ms_per_keystroke: number }
type Suite = { rows: { test: number; calib: number }; rivals: Record<string, { test: Split; calib: Split; latency_ms: number | null }>; stream?: Stream }
type System = { name: string; spec: { description: string; nodes: Record<string, { specialist: string; role: string }> }; board: Record<string, Suite> | null; exported: boolean }

const config = {
  exact_frame: { label: "exact frame", color: "var(--chart-1)" },
  action_accuracy: { label: "action", color: "var(--chart-2)" },
  arg_f1: { label: "arguments F1", color: "var(--chart-3)" },
} satisfies ChartConfig

const pct = (x: number) => `${Math.round(x * 100)}%`

function SuiteCard({ name, suite }: { name: string; suite: Suite }) {
  const data = Object.entries(suite.rivals).map(([rival, v]) => ({ rival, ...v.test }))
  const st = suite.stream
  return (
    <Card>
      <CardHeader>
        <CardTitle>{name}</CardTitle>
        <CardDescription>
          {suite.rows.test} test requests, scored end to end against the LLM doing the whole task.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ChartContainer config={config} className="h-64 w-full">
          <BarChart data={data}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="rival" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} tickLine={false} axisLine={false} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            {Object.keys(config).map((k) => (
              <Bar key={k} dataKey={k} fill={`var(--color-${k})`} radius={4} />
            ))}
          </BarChart>
        </ChartContainer>
        {st?.gate && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              ["typed when the action fires", pct(st.test.typed_share)],
              ["fired before the last key", pct(st.test.early_share)],
              ["wrong per 100", `${st.test.wrong_per_100.toFixed(1)} (end ${st.test_at_end.wrong_per_100.toFixed(1)})`],
              ["per keystroke", `${st.ms_per_keystroke.toFixed(1)} ms`],
            ].map(([label, value]) => (
              <Card key={label}>
                <CardHeader className="pb-2">
                  <CardDescription>{label}</CardDescription>
                  <CardTitle className="font-mono text-xl">{value}</CardTitle>
                </CardHeader>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function Systems() {
  const [items, setItems] = useState<System[]>([])
  useEffect(() => {
    api<System[]>("/systems").then(setItems)
  }, [])
  return (
    <div className="space-y-6">
      {items.map((s) => (
        <div key={s.name} className="space-y-3">
          <div>
            <h2 className="text-xl font-semibold">{s.name}</h2>
            <p className="text-muted-foreground text-sm">{s.spec.description}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {Object.entries(s.spec.nodes).map(([k, n]) => (
                <Badge key={k} variant="secondary">
                  {n.role}: {n.specialist}
                </Badge>
              ))}
            </div>
          </div>
          {s.exported && <TypeDemo system={s.name} />}
          {Object.entries(s.board ?? {}).map(([name, suite]) => (
            <SuiteCard key={name} name={name} suite={suite} />
          ))}
        </div>
      ))}
    </div>
  )
}
