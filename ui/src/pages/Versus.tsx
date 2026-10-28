import { Loader2 } from "lucide-react"
import { useEffect, useState } from "react"
import { api, type Detail } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { signed, signedInterval } from "@/format"
import { ForestChart } from "./ForestChart"

type Lead = { delta: number; lo: number; hi: number; p_better: number } | null
type Paired = { a: string; b: string; sure: number; suites: { suite: string; calib: Lead; test: Lead }[] }

const calibOf = (d: Detail, m: string) => d.board[m]?.[d.primary_suite]?.calib_objective ?? -1

function Verdict({ lead, sure }: { lead: Lead; sure: number }) {
  if (!lead) return <span className="text-muted-foreground">-</span>
  if (lead.p_better >= sure) return <Badge>sure lead</Badge>
  if (lead.p_better <= 1 - sure) return <Badge variant="destructive">sure loss</Badge>
  return <Badge variant="outline">not sure</Badge>
}

function Pick({ value, models, onChange, label }: { value: string; models: string[]; onChange: (v: string) => void; label: string }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-44" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {models.map((m) => (
          <SelectItem key={m} value={m}>
            {m}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export function Versus({ d }: { d: Detail }) {
  const models = Object.keys(d.board).sort((x, y) => calibOf(d, y) - calibOf(d, x))
  const [a, setA] = useState(d.promoted ?? models[1] ?? "")
  const [b, setB] = useState(models.find((m) => m !== (d.promoted ?? models[1])) ?? "")
  const [paired, setPaired] = useState<Paired | null>(null)
  useEffect(() => {
    if (!a || !b) return
    setPaired(null)
    api<Paired>(`/specialists/${d.name}/versus?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`).then(setPaired, () => setPaired(null))
  }, [d.name, a, b])
  if (models.length < 2) return null
  const rows = (paired?.suites ?? []).flatMap((s) => (s.test ? [{ title: s.suite, mean: s.test.delta, lo: s.test.lo, hi: s.test.hi }] : []))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Compare two models
          <InfoTip>
            Paired bootstrap on the same rows of each suite: the second model minus the first, with a 95% interval. Calib decides a lead; test is only reported.
          </InfoTip>
        </CardTitle>
        <CardAction className="flex items-center gap-2">
          <Pick value={a} models={models.filter((m) => m !== b)} onChange={setA} label="First model" />
          <span className="text-muted-foreground text-sm">vs</span>
          <Pick value={b} models={models.filter((m) => m !== a)} onChange={setB} label="Second model" />
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        {!paired ? (
          <Loader2 className="text-muted-foreground size-4 animate-spin" />
        ) : (
          <>
            {rows.length > 0 && <ForestChart rows={rows} label={`Test, ${b} minus ${a}`} readout={(r) => `${signed(r.mean)} ${signedInterval(r.lo, r.hi)}`} />}
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Suite</TableHead>
                  <TableHead className="text-right">Calib lead</TableHead>
                  <TableHead>On calib</TableHead>
                  <TableHead className="text-right">Test lead</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paired.suites.map((s) => (
                  <TableRow key={s.suite}>
                    <TableCell className="font-medium">{s.suite}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {s.calib ? `${signed(s.calib.delta)} ${signedInterval(s.calib.lo, s.calib.hi)}` : "-"}
                    </TableCell>
                    <TableCell>
                      <Verdict lead={s.calib} sure={paired.sure} />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {s.test ? `${signed(s.test.delta)} ${signedInterval(s.test.lo, s.test.hi)}` : "-"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </>
        )}
      </CardContent>
    </Card>
  )
}
