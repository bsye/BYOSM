import { Loader2, Search } from "lucide-react"
import { useEffect, useState } from "react"
import { api, post } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { RiskDrivers, type Risk } from "./RiskDrivers"

type Inputs = { key: string | null; columns: string[] }

const LOOKUP = "lookup"
const FILL = "fill"

function Result({ name, request, answer, shown }: { name: string; request: object; answer: Risk; shown: string[] }) {
  const who = shown
    .map((c) => answer.row[c])
    .filter(Boolean)
    .join(" · ")
  if (answer.state !== "alive")
    return (
      <p className="text-sm">
        <Badge variant="outline">{answer.state === "ended" ? "Already ended" : "Not scored"}</Badge> <span className="text-muted-foreground">{who}</span>
      </p>
    )
  return <RiskDrivers name={name} request={request} base={answer} who={who} />
}

export function ForecastTry({ name, shown }: { name: string; shown: string[] }) {
  const [inputs, setInputs] = useState<Inputs | null>(null)
  const [mode, setMode] = useState(LOOKUP)
  const [key, setKey] = useState("")
  const [row, setRow] = useState<Record<string, string>>({})
  const [answer, setAnswer] = useState<Risk | null>(null)
  const [request, setRequest] = useState<object>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  useEffect(() => {
    api<Inputs>(`/forecasts/${name}/inputs`).then(setInputs, () => setInputs(null))
  }, [name])
  const ask = () => {
    setBusy(true)
    setError("")
    const body = mode === LOOKUP ? { key } : { row }
    post<Risk>(`/forecasts/${name}/predict`, body)
      .then(
        (r) => {
          setRequest({ row: r.row })
          setAnswer(r)
        },
        (e) => setError(String(e)),
      )
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
        {answer && <Result key={JSON.stringify(request)} name={name} request={request} answer={answer} shown={shown} />}
      </CardContent>
    </Card>
  )
}
