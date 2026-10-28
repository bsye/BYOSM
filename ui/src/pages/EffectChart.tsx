import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis, type BarShapeProps } from "recharts"
import { axisWidth, ROW_PX } from "@/chart-size"
import { InfoTip } from "@/components/info-tip"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { signed, signedInterval } from "@/format"

type Summary = { measured: number; mean: number | null; lo: number | null; hi: number | null }
export type Effect = Summary & { left_out: number; kinds: (Summary & { modality: string; task_type: string })[] }
type Family = { title: string; calib_effect: Effect }
type Row = { title: string; mean: number; lo: number; hi: number; range: [number, number]; effect: Effect }

const DOT_R = 4

const config = { range: { label: "Calib effect", color: "var(--chart-2)" } } satisfies ChartConfig

const rowOf = (f: Family): Row | null => {
  const { mean, lo, hi } = f.calib_effect
  return mean == null || lo == null || hi == null ? null : { title: f.title, mean, lo, hi, range: [lo, hi], effect: f.calib_effect }
}

function Interval({ x, y, width, height, payload, isActive }: BarShapeProps) {
  const row = payload as Row
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

function ReadoutRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex w-full justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground font-mono font-medium tabular-nums">{value}</span>
    </div>
  )
}

function Readout({ effect }: { effect: Effect }) {
  return (
    <div className="grid w-full gap-1">
      <ReadoutRow label="Mean" value={signed(effect.mean)} />
      <ReadoutRow label="95% interval" value={signedInterval(effect.lo, effect.hi)} />
      <ReadoutRow label="Experiments" value={String(effect.measured)} />
      {effect.left_out > 0 && <ReadoutRow label="Left out" value={String(effect.left_out)} />}
      {effect.kinds.length > 1 &&
        effect.kinds.map((k) => (
          <ReadoutRow key={`${k.modality}-${k.task_type}`} label={`${k.modality} ${k.task_type}`} value={`${signed(k.mean)} (${k.measured})`} />
        ))}
    </div>
  )
}

function Forest({ rows }: { rows: Row[] }) {
  return (
    <ChartContainer config={config} className="w-full" style={{ height: rows.length * ROW_PX + 60 }}>
      <BarChart accessibilityLayer data={rows} layout="vertical" margin={{ left: 8, right: 16 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" domain={["auto", "auto"]} tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(v) => signed(Number(v))} />
        <YAxis type="category" dataKey="title" tickLine={false} axisLine={false} width={axisWidth(rows.map((r) => r.title))} interval={0} />
        <ReferenceLine x={0} ifOverflow="extendDomain" stroke="var(--muted-foreground)" strokeDasharray="4 4" />
        <ChartTooltip
          cursor={false}
          content={<ChartTooltipContent hideIndicator formatter={(_value, _name, item) => <Readout effect={(item.payload as Row).effect} />} />}
        />
        <Bar dataKey="range" fill="var(--color-range)" shape={Interval} isAnimationActive={false} />
      </BarChart>
    </ChartContainer>
  )
}

export function EffectChart({ families }: { families: Family[] }) {
  const rows = families
    .map(rowOf)
    .filter((r): r is Row => r != null)
    .sort((a, b) => b.mean - a.mean)
  const few = families.filter((f) => f.calib_effect.lo == null && f.calib_effect.measured + f.calib_effect.left_out > 0)
  if (!rows.length && !few.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Calib effect per family
          <InfoTip>
            Mean change in calib score against the experiment each one was built from, scored on the same suite version, with a 95% Student t interval across
            experiments. Experiments without such a parent are left out.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      {rows.length > 0 && (
        <CardContent>
          <Forest rows={rows} />
        </CardContent>
      )}
      {few.length > 0 && (
        <CardFooter className="text-muted-foreground text-xs">
          Not enough runs: {few.map((f) => `${f.title} (${f.calib_effect.measured} measured)`).join(", ")}
        </CardFooter>
      )}
    </Card>
  )
}
