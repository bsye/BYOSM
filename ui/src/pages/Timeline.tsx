import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from "recharts"
import type { Job } from "@/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"
import { useNow } from "@/now"

const SHOWN = 12
const ROW_PX = 28
const STATUS_COLOR: Record<Job["status"], string> = {
  queued: "var(--muted-foreground)",
  running: "var(--chart-1)",
  done: "var(--chart-2)",
  failed: "var(--destructive)",
  cancelled: "var(--muted-foreground)",
}

const config = { span: { label: "running" } } satisfies ChartConfig

const clock = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

export function Timeline({ jobs }: { jobs: Job[] }) {
  const recent = jobs.filter((j) => j.started != null).slice(0, SHOWN).reverse()
  const now = useNow(recent.some((j) => j.status === "running"))
  if (!recent.length) return null
  const data = recent.map((j) => ({
    name: `${j.args.slice(0, 2).join(" ")} ${j.id.slice(0, 4)}`,
    span: [j.started as number, j.ended ?? now],
    status: j.status,
    command: `byosm ${j.args.join(" ")}`,
  }))
  const start = Math.min(...data.map((d) => d.span[0]))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Timeline</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="w-full" style={{ height: data.length * ROW_PX + 40 }}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
            <CartesianGrid horizontal={false} />
            <XAxis type="number" domain={[start, "dataMax"]} tickFormatter={clock} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="name" width={150} tickLine={false} axisLine={false} interval={0} />
            <ChartTooltip
              content={({ payload }) => {
                const d = payload?.[0]?.payload
                if (!d) return null
                return (
                  <div className="bg-background rounded-md border px-3 py-2 text-xs shadow-sm">
                    <div className="font-mono">{d.command}</div>
                    <div className="text-muted-foreground">
                      {d.status}, {clock(d.span[0])} to {clock(d.span[1])}
                    </div>
                  </div>
                )
              }}
            />
            <Bar dataKey="span" radius={4} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.name} fill={STATUS_COLOR[d.status]} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
