import { Loader2, Shuffle } from "lucide-react"
import { useEffect, useState } from "react"
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, XAxis, YAxis } from "recharts"
import { api, post } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { score } from "@/format"
import type { ForecasterView } from "./Forecaster"

type Example = { config: string; freq: string; item: string; history: (number | null)[]; future: (number | null)[] }
type Forecast = { version: string; horizon: number; quantiles: number[]; forecast: number[][] }

const SHOWN_HORIZONS = 4
const LOW = 0.1
const MID = 0.5
const HIGH = 0.9

const chart = {
  history: { label: "History", color: "var(--muted-foreground)" },
  actual: { label: "Actual", color: "var(--chart-2)" },
  median: { label: "Forecast", color: "var(--chart-1)" },
  band: { label: "80% interval", color: "var(--chart-1)" },
} satisfies ChartConfig

function rows(history: (number | null)[], future: (number | null)[], f: Forecast | null) {
  const h = f?.horizon ?? future.length
  const past = history.slice(-SHOWN_HORIZONS * h)
  const at = (q: number) => f?.forecast[f.quantiles.indexOf(q)] ?? []
  const [low, mid, high] = [at(LOW), at(MID), at(HIGH)]
  return [
    ...past.map((v, i) => ({ t: i - past.length, history: v })),
    ...Array.from({ length: Math.max(h, future.length) }, (_, i) => ({
      t: i,
      actual: future[i] ?? undefined,
      median: mid[i],
      band: low[i] == null ? undefined : [low[i], high[i]],
    })),
  ]
}

function meanAbsError(future: (number | null)[], f: Forecast | null) {
  const mid = f?.forecast[f.quantiles.indexOf(MID)] ?? []
  const pairs = future.flatMap((v, i) => (v == null || mid[i] == null ? [] : [Math.abs(v - mid[i])]))
  return pairs.length ? pairs.reduce((a, b) => a + b, 0) / pairs.length : null
}

function parseValues(text: string): (number | null)[] {
  return text
    .split("\n")
    .flatMap((line) => line.split(","))
    .map((v) => v.trim())
    .filter((v) => v.length)
    .map((v) => (Number.isFinite(Number(v)) ? Number(v) : null))
}

export function SeriesTry({ f }: { f: ForecasterView }) {
  const configs = Object.keys(f.configs ?? {})
  const [config, setConfig] = useState(configs[0] ?? "")
  const [index, setIndex] = useState(0)
  const [example, setExample] = useState<Example | null>(null)
  const [forecast, setForecast] = useState<Forecast | null>(null)
  const [pasted, setPasted] = useState("")
  const [own, setOwn] = useState<Forecast | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!f.live || !config) return
    setExample(null)
    setForecast(null)
    api<Example>(`/series/${f.name}/example?config=${encodeURIComponent(config)}&index=${index}`).then((e) => {
      setExample(e)
      post<Forecast>(`/series/${f.name}/forecast`, { history: e.history, freq: e.freq }).then(setForecast)
    })
  }, [f.name, f.live, config, index])

  if (!f.live)
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No saved model yet</EmptyTitle>
          <EmptyDescription>Keep training saves the model of every new version, then you can forecast with it here.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )

  const values = parseValues(pasted)
  const runOwn = () => {
    setBusy(true)
    post<Forecast>(`/series/${f.name}/forecast`, { history: values, freq: example?.freq ?? "D" })
      .then(setOwn)
      .finally(() => setBusy(false))
  }
  const mae = meanAbsError(example?.future ?? [], forecast)

  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle>{example ? `${example.config} · ${example.item}` : "Test window"}</CardTitle>
          <div className="flex items-center gap-2">
            <Select value={config} onValueChange={setConfig}>
              <SelectTrigger className="w-56" aria-label="Dataset">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {configs.map((c) => (
                  <SelectItem key={c} value={c}>
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" onClick={() => setIndex((i) => i + 1)}>
              <Shuffle />
              Another series
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-2">
          {example ? (
            <ChartContainer config={chart} className="h-72 w-full">
              <ComposedChart data={rows(example.history, example.future, forecast)} margin={{ top: 8, left: 8, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="t" tickLine={false} axisLine={false} minTickGap={32} />
                <YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v) => Number(v).toPrecision(3)} />
                <ReferenceLine x={0} strokeDasharray="4 4" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Area dataKey="band" stroke="none" fill="var(--color-band)" fillOpacity={0.2} isAnimationActive={false} />
                <Line dataKey="history" stroke="var(--color-history)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line dataKey="actual" stroke="var(--color-actual)" strokeDasharray="4 3" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line dataKey="median" stroke="var(--color-median)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </ComposedChart>
            </ChartContainer>
          ) : (
            <div className="text-muted-foreground flex h-72 items-center justify-center gap-2 text-sm">
              <Loader2 className="size-4 animate-spin" />
              Loading a test window
            </div>
          )}
          <p className="text-muted-foreground text-sm">
            {forecast
              ? `${forecast.version}${forecast.version === f.promoted ? "" : `, newest saved (${f.promoted} is promoted)`} · mean absolute error ${mae == null ? "-" : score(mae)}`
              : example
                ? "Forecasting"
                : ""}
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Your series</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Label htmlFor="own-series">{`Values, oldest first, one per line or comma separated (${example?.freq ?? "same"} frequency)`}</Label>
          <Textarea id="own-series" rows={4} value={pasted} onChange={(e) => setPasted(e.target.value)} placeholder="12, 15, 14, 18, 21, ..." />
          <div>
            <Button onClick={runOwn} disabled={values.length < 2 || busy}>
              {busy && <Loader2 className="animate-spin" />}
              Forecast
            </Button>
          </div>
          {own && (
            <ChartContainer config={chart} className="h-64 w-full">
              <ComposedChart data={rows(values, [], own)} margin={{ top: 8, left: 8, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="t" tickLine={false} axisLine={false} minTickGap={32} />
                <YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v) => Number(v).toPrecision(3)} />
                <ReferenceLine x={0} strokeDasharray="4 4" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Area dataKey="band" stroke="none" fill="var(--color-band)" fillOpacity={0.2} isAnimationActive={false} />
                <Line dataKey="history" stroke="var(--color-history)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line dataKey="median" stroke="var(--color-median)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </ComposedChart>
            </ChartContainer>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
