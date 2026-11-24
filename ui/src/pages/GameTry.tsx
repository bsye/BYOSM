import { FileText, Info, Loader2, Search } from "lucide-react"
import { useState } from "react"
import { post } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { percent, type Advice } from "./game-api"
import { EMPTY, EXAMPLE, history, ready, type Draft } from "./hand-builder"
import { HandBuilder } from "./HandBuilder"

const BUILD = "build"
const PASTE = "paste"
const STUDENT = "student"
const SHOWN_FROM = 0.001
const FORMAT =
  "Amounts are chips; a raise is the total put in on that street. The student needs six players with equal stacks; SHARINGAN-2 takes three to six. Moves played less than 0.1% of the time are left out."

function Moves({ advice }: { advice: Advice }) {
  const moves = advice.moves.filter((m) => m.probability >= SHOWN_FROM).sort((a, b) => b.probability - a.probability)
  return (
    <div className="grid gap-2">
      <p className="text-muted-foreground text-sm">
        {advice.player} to act on the {advice.street} · <span className="tabular-nums">{advice.ms.toFixed(0)} ms</span>
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Move</TableHead>
            <TableHead className="w-2/3">How often</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {moves.map((m) => (
            <TableRow key={m.move}>
              <TableCell className="font-medium">{m.move}</TableCell>
              <TableCell>
                <div className="flex items-center gap-3">
                  <Progress value={m.probability * 100} />
                  <span className="w-14 shrink-0 text-right tabular-nums">{percent(m.probability)}</span>
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

export function GameTry({ name, blueprints }: { name: string; blueprints: string[] }) {
  const [mode, setMode] = useState(BUILD)
  const [advisor, setAdvisor] = useState(blueprints[0] ?? STUDENT)
  const [draft, setDraft] = useState<Draft>(EMPTY)
  const [pasted, setPasted] = useState("")
  const [advice, setAdvice] = useState<Advice | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const building = mode === BUILD
  const ask = () => {
    setBusy(true)
    setError("")
    const path = advisor === STUDENT ? `/games/${encodeURIComponent(name)}/advise` : `/blueprints/${encodeURIComponent(advisor)}/advise`
    post<Advice>(path, { hand: building ? history(draft) : pasted })
      .then(setAdvice, (e) => {
        setAdvice(null)
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setBusy(false))
  }
  const example = () => {
    setDraft(EXAMPLE)
    setPasted(history(EXAMPLE))
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Try it
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="text-muted-foreground size-4" />
            </TooltipTrigger>
            <TooltipContent>{FORMAT}</TooltipContent>
          </Tooltip>
        </CardTitle>
        <CardAction className="flex flex-wrap justify-end gap-2">
          {blueprints.length > 0 && (
            <ToggleGroup type="single" size="sm" variant="outline" value={advisor} onValueChange={(v) => v && setAdvisor(v)} aria-label="Who answers">
              {blueprints.map((b) => (
                <ToggleGroupItem key={b} value={b} className="uppercase">
                  {b}
                </ToggleGroupItem>
              ))}
              <ToggleGroupItem value={STUDENT}>Student</ToggleGroupItem>
            </ToggleGroup>
          )}
          <ToggleGroup type="single" size="sm" variant="outline" value={mode} onValueChange={(v) => v && setMode(v)}>
            <ToggleGroupItem value={BUILD}>Build</ToggleGroupItem>
            <ToggleGroupItem value={PASTE}>Paste</ToggleGroupItem>
          </ToggleGroup>
          <Button size="sm" variant="outline" onClick={example}>
            <FileText />
            Example
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-4">
        {building ? (
          <HandBuilder draft={draft} onChange={setDraft} />
        ) : (
          <Textarea
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            placeholder="Paste a PHH hand history that stops at the decision"
            className="max-h-64 min-h-40 font-mono text-xs"
            spellCheck={false}
          />
        )}
        <div className="flex justify-end">
          <Button size="sm" disabled={busy || (building ? !ready(draft) : !pasted.trim())} onClick={ask}>
            {busy ? <Loader2 className="animate-spin" /> : <Search />}
            Advise
          </Button>
        </div>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {advice && <Moves advice={advice} />}
      </CardContent>
    </Card>
  )
}
