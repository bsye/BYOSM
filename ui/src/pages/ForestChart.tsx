import type { ReactNode } from "react"
import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis, type BarShapeProps } from "recharts"
import { axisWidth, ROW_PX } from "@/chart-size"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { signed } from "@/format"

export type ForestRow = { title: string; mean: number; lo: number; hi: number }

const DOT_R = 4
const AXIS_ROOM_PX = 60

function Interval({ x, y, width, height, payload, isActive }: BarShapeProps) {
  const row = payload as ForestRow
  const span = row.hi - row.lo
  const cx = x + (span ? (width * (row.mean - row.lo)) / span : 0)
  const cy = y + height / 2
  return (
    <g>
      <line x1={x} x2={x + width} y1={cy} y2={cy} stroke="var(--color-range)" strokeWidth={2} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={isActive ? DOT_R + 1 : DOT_R} fill="var(--color-range)" stroke="var(--card)" strokeWidth={2} />
    </g>
  )
}

export function ForestChart<T extends ForestRow>({ rows, label, readout }: { rows: T[]; label: string; readout: (row: T) => ReactNode }) {
  const config = { range: { label, color: "var(--chart-2)" } } satisfies ChartConfig
  const data = rows.map((r) => ({ ...r, range: [r.lo, r.hi] }))
  return (
    <ChartContainer config={config} className="w-full" style={{ height: rows.length * ROW_PX + AXIS_ROOM_PX }}>
      <BarChart accessibilityLayer data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" domain={["auto", "auto"]} tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v) => signed(Number(v))} />
        <YAxis type="category" dataKey="title" tickLine={false} axisLine={false} width={axisWidth(rows.map((r) => r.title))} interval={0} />
        <ReferenceLine x={0} ifOverflow="extendDomain" stroke="var(--muted-foreground)" strokeDasharray="4 4" />
        <ChartTooltip cursor={false} content={<ChartTooltipContent hideIndicator formatter={(_v, _n, item) => readout(item.payload as T)} />} />
        <Bar dataKey="range" fill="var(--color-range)" shape={Interval} isAnimationActive={false} />
      </BarChart>
    </ChartContainer>
  )
}
