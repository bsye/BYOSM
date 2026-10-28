import { Loader2 } from "lucide-react"
import { Area, AreaChart, CartesianGrid, ReferenceArea, XAxis, YAxis } from "recharts"
import { InfoTip } from "@/components/info-tip"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { pct, useWordSwitches } from "./word-switches"
import { WordSwitch } from "./WordSwitch"

type Span = { label: string; text: string; start: number; end: number }
type Tagged = { words: string[]; tags: string[]; confidence: number[]; entities: Span[] }

const TONES = ["var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"]
const OTHER_TONE = "var(--muted-foreground)"
const SPAN_TINT = 30
const AREA_OPACITY = 0.12
const EDGE = 0.45
const config = { p: { label: "Confidence", color: "var(--chart-2)" } } satisfies ChartConfig

const kindOf = (tag: string | undefined) => (!tag || tag === "O" ? null : tag.split("-").slice(1).join("-"))

function toneOf(kinds: string[], kind: string): string {
  const at = kinds.indexOf(kind)
  return at >= 0 && at < TONES.length ? TONES[at] : OTHER_TONE
}

function placed(now: Tagged, kept: number[]) {
  const at = new Map(kept.map((orig, j) => [orig, j]))
  const spans = now.entities.map((e) => ({ ...e, from: kept[e.start], to: kept[e.end - 1] }))
  return { at, spans, ends: new Map(spans.map((s) => [s.to, s.label])) }
}

export function TaggingView({ name, text }: { name: string; text: string }) {
  const { base, now, off, kept, removed, busy, toggle } = useWordSwitches<Tagged>(`/specialists/${name}/predict`, text)
  if (!base || !now) return busy ? <Loader2 className="text-muted-foreground animate-spin" /> : null
  const kinds = [...new Set(base.entities.map((e) => e.label))]
  const { at, spans, ends } = placed(now, kept)
  const data = base.words.map((word, i) => {
    const j = at.get(i)
    return { i, word, tag: j == null ? "off" : now.tags[j], p: j == null ? undefined : now.confidence[j] }
  })
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          How it tags
          <InfoTip>
            The line is how sure the model is of the tag it gives each word. Each found span is shaded by its kind. Click a word to switch it off and tag the
            input again without it.
          </InfoTip>
        </CardTitle>
        <CardDescription className="tabular-nums">
          {removed.length
            ? `Without ${removed.map((w) => `"${w}"`).join(", ")}: ${now.entities.length} spans, ${base.entities.length} before`
            : "Click a word to switch it off"}
        </CardDescription>
        {busy && (
          <CardAction>
            <Loader2 className="text-muted-foreground size-4 animate-spin" />
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <ChartContainer config={config} className="h-48 w-full">
          <AreaChart data={data} margin={{ left: 8, right: 8, top: 8 }}>
            <CartesianGrid vertical={false} />
            {spans.map((s) => (
              <ReferenceArea key={`${s.from}-${s.label}`} x1={s.from - EDGE} x2={s.to + EDGE} fill={toneOf(kinds, s.label)} fillOpacity={AREA_OPACITY} />
            ))}
            <XAxis
              dataKey="i"
              type="number"
              domain={[-EDGE, base.words.length - 1 + EDGE]}
              ticks={data.map((d) => d.i)}
              tickFormatter={(i) => base.words[Number(i)] ?? ""}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
              minTickGap={16}
            />
            <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={pct} tickLine={false} axisLine={false} width={40} />
            <ChartTooltip
              content={
                <ChartTooltipContent labelFormatter={(_, items) => `${items[0]?.payload.word} · ${items[0]?.payload.tag}`} formatter={(v) => pct(Number(v))} />
              }
            />
            <Area dataKey="p" type="linear" stroke="var(--color-p)" strokeWidth={2} fill="var(--color-p)" fillOpacity={0.08} isAnimationActive={false} />
          </AreaChart>
        </ChartContainer>
        <div className="flex flex-wrap gap-1 leading-7">
          {base.words.map((word, i) => {
            const j = at.get(i)
            const kind = j == null ? null : kindOf(now.tags[j])
            return (
              <WordSwitch
                key={`${i}-${word}`}
                word={word}
                off={off.includes(i)}
                tone={kind ? `color-mix(in oklch, ${toneOf(kinds, kind)} ${SPAN_TINT}%, transparent)` : undefined}
                title={j == null ? "switched off" : `${now.tags[j]} ${pct(now.confidence[j])}`}
                onClick={() => toggle(i)}
              >
                {ends.has(i) && <span className="text-muted-foreground ml-1 text-xs">{ends.get(i)}</span>}
              </WordSwitch>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
