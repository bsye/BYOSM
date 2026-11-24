import { FileText, Info, Loader2, Search } from "lucide-react"
import { useState } from "react"
import { post } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { percent, type Advice } from "./game-api"

const EXAMPLE = `variant = 'NT'
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 4sAd', 'd dh p2 KsQh', 'd dh p3 8h2c', 'd dh p4 6c4h', 'd dh p5 JcQd', 'd dh p6 Kc9s', 'p3 f', 'p4 f', 'p5 cbr 250', 'p6 f', 'p1 f', 'p2 cc', 'd db 4d6sTh', 'p2 cc', 'p5 cbr 300']
players = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
`
const FORMAT = "A PHH hand history that stops at the decision: six players, equal stacks, and the cards of the player to act."

function Moves({ advice }: { advice: Advice }) {
  const moves = [...advice.moves].sort((a, b) => b.probability - a.probability)
  return (
    <div className="grid gap-2">
      <p className="text-muted-foreground text-sm">
        Seat {advice.seat} to act on the {advice.street} · <span className="tabular-nums">{advice.ms.toFixed(0)} ms</span>
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

export function GameTry({ name }: { name: string }) {
  const [hand, setHand] = useState("")
  const [advice, setAdvice] = useState<Advice | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const ask = () => {
    setBusy(true)
    setError("")
    post<Advice>(`/games/${encodeURIComponent(name)}/advise`, { hand })
      .then(setAdvice, (e) => {
        setAdvice(null)
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setBusy(false))
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
        <CardAction>
          <Button size="sm" variant="outline" onClick={() => setHand(EXAMPLE)}>
            <FileText />
            Example
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-4">
        <Textarea
          value={hand}
          onChange={(e) => setHand(e.target.value)}
          placeholder="Paste a hand history"
          className="max-h-64 min-h-40 font-mono text-xs"
          spellCheck={false}
        />
        <div className="flex justify-end">
          <Button size="sm" disabled={!hand.trim() || busy} onClick={ask}>
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
