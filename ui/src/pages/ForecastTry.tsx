import { Loader2, Search } from "lucide-react"
import { useEffect, useState } from "react"
import { api, post } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

type Inputs = { key: string | null; columns: string[] }
type Reason = { feature: string; drop: number; value: string | number | null }
type Answer = { state: "alive" | "ended"; row: Record<string, string | null>; chance?: number; why?: Reason[] }

const LOOKUP = "lookup"
const FILL = "fill"
const NEVER_DAYS = 9999
const pct = (v: number) => `${Math.round(v * 100)}%`
const value = (v: string | number | null) => (v == null ? "missing" : v === NEVER_DAYS ? "never" : String(v))

function Result({ answer, shown }: { answer: Answer; shown: string[] }) {
  const who = shown
    .map((c) => answer.row[c])
    .filter(Boolean)
    .join(" · ")
  if (answer.state === "ended")
    return (
      <p className="text-sm">
        <Badge variant="outline">Already ended</Badge> <span className="text-muted-foreground">{who}</span>
      </p>
    )
  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="text-3xl font-semibold tabular-nums">{pct(answer.chance ?? 0)}</span>
        <span className="text-muted-foreground text-sm">{who}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {(answer.why ?? []).map((w) => (
          <Badge key={w.feature} variant="secondary" className="font-mono" title={`+${pct(w.drop)} because of this`}>
            {w.feature} {value(w.value)}
          </Badge>
        ))}
      </div>
    </div>
  )
}

export function ForecastTry({ name, shown }: { name: string; shown: string[] }) {
  const [inputs, setInputs] = useState<Inputs | null>(null)
  const [mode, setMode] = useState(LOOKUP)
  const [key, setKey] = useState("")
  const [row, setRow] = useState<Record<string, string>>({})
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  useEffect(() => {
    api<Inputs>(`/forecasts/${name}/inputs`).then(setInputs, () => setInputs(null))
  }, [name])
  const ask = () => {
    setBusy(true)
    setError("")
    post<Answer>(`/forecasts/${name}/predict`, mode === LOOKUP ? { key } : { row })
      .then(setAnswer, (e) => setError(String(e)))
      .finally(() => setBusy(false))
  }
  const ready = mode === LOOKUP ? key.trim() : Object.values(row).some(Boolean)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Try it</CardTitle>
        <CardAction>
          <ToggleGroup type="single" size="sm" variant="outline" value={mode} onValueChange={(v) => v && setMode(v)}>
            <ToggleGroupItem value={LOOKUP}>Look up</ToggleGroupItem>
            <ToggleGroupItem value={FILL}>Fill in</ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-4">
        {mode === LOOKUP ? (
          <div className="flex gap-2">
            <Input
              value={key}
              placeholder={inputs?.key ?? "key"}
              onChange={(e) => setKey(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ready && ask()}
            />
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {(inputs?.columns ?? []).map((c) => (
              <div key={c} className="grid gap-1.5">
                <Label className="font-mono text-xs">{c}</Label>
                <Input value={row[c] ?? ""} onChange={(e) => setRow({ ...row, [c]: e.target.value })} />
              </div>
            ))}
          </div>
        )}
        <div className="flex justify-end">
          <Button size="sm" disabled={!ready || busy} onClick={ask}>
            {busy ? <Loader2 className="animate-spin" /> : <Search />}
            Predict
          </Button>
        </div>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {answer && <Result answer={answer} shown={shown} />}
      </CardContent>
    </Card>
  )
}
