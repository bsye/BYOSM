import { Loader2 } from "lucide-react"
import { Area, AreaChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts"
import { InfoTip } from "@/components/info-tip"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { pct, useWordSwitches } from "./word-switches"
import { WordSwitch } from "./WordSwitch"

type Reading = {
  label: string
  follows: string
  words: string[]
  probs: Record<string, number>
  reading: { word: string; at: number; p: number }[]
  drivers: { word: string; at: number; drop: number }[]
  threshold: number | null
}

const SHADE_MAX = 85

function shade(drop: number): string {
  const strength = Math.min(SHADE_MAX, Math.round(Math.abs(drop) * 100))
  const tone = drop >= 0 ? "var(--chart-2)" : "var(--chart-4)"
  return `color-mix(in oklch, ${tone} ${strength}%, transparent)`
}

export function ReadingView({ name, text }: { name: string; text: string }) {
  const { base, now, off, removed, busy, toggle } = useWordSwitches<Reading>(`/specialists/${name}/explain`, text, (b) => ({ follow: b.label }))
  if (!base || !now) return busy ? <Loader2 className="text-muted-foreground animate-spin" /> : null
  const before = base.probs[base.label] ?? 0
  const after = now.probs[base.label] ?? 0
  const config = { p: { label: base.label, color: "var(--chart-2)" } } satisfies ChartConfig
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          How it reads
          <InfoTip>
            The line is the probability of the answer after each word it reads. A shaded word moves the answer: the darker, the more the probability falls
            without it. Click a word to switch it off and read the input again without it.
          </InfoTip>
        </CardTitle>
        <CardDescription className="tabular-nums">
          {removed.length
            ? `Without ${removed.map((w) => `"${w}"`).join(", ")}: ${base.label} ${pct(before)} → ${pct(after)}${now.label !== base.label ? `, now ${now.label}` : ""}`
            : "Click a word to switch it off"}
        </CardDescription>
        {busy && (
          <CardAction>
            <Loader2 className="text-muted-foreground size-4 animate-spin" />
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <ChartContainer config={config} className="h-56 w-full">
          <AreaChart data={now.reading} margin={{ left: 8, right: 8, top: 8 }}>
            <defs>
              <linearGradient id="reading-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-p)" stopOpacity={0.35} />
                <stop offset="95%" stopColor="var(--color-p)" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="word" tickLine={false} axisLine={false} interval="preserveStartEnd" minTickGap={16} />
            <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={pct} tickLine={false} axisLine={false} width={40} />
            {now.threshold != null && <ReferenceLine y={now.threshold} stroke="var(--muted-foreground)" strokeDasharray="4 4" />}
            <ChartTooltip content={<ChartTooltipContent formatter={(v) => pct(Number(v))} />} />
            <Area dataKey="p" type="monotone" stroke="var(--color-p)" strokeWidth={2} fill="url(#reading-fill)" isAnimationActive={false} />
          </AreaChart>
        </ChartContainer>
        <div className="flex flex-wrap gap-1 leading-7">
          {base.drivers.map((d, i) => (
            <WordSwitch
              key={`${i}-${d.word}`}
              word={d.word}
              off={off.includes(i)}
              tone={shade(d.drop)}
              title={`${d.drop >= 0 ? "-" : "+"}${pct(Math.abs(d.drop))} without it`}
              onClick={() => toggle(i)}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
