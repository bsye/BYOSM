import { ListFilter } from "lucide-react"
import { useState } from "react"
import type { Detail } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { ModelName } from "@/components/model-name"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ms, params, rivalName, score } from "@/format"
import { BoardChart, boardRows } from "./BoardChart"
import { SizeChart } from "./SizeChart"

const FOCUS_OTHERS = 5

const display = (d: Detail, model: string) => (d.baselines.includes(model) ? rivalName(model) : model)

export function Board({ d }: { d: Detail }) {
  const [all, setAll] = useState(false)
  const every = boardRows(d.board, d.primary_suite)
  const keep = new Set([d.promoted, ...d.baselines])
  const others = every
    .filter((r) => !keep.has(r.model))
    .slice(0, FOCUS_OTHERS)
    .map((r) => r.model)
  const rows = all ? every : every.filter((r) => keep.has(r.model) || others.includes(r.model))
  const caughtLabels = Object.keys(d.board[d.promoted ?? ""]?.[d.primary_suite]?.caught ?? {})
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Test {d.metric} on {d.benchmark}
          <InfoTip>Test with its 95% interval, next to calib. Calib picks the model; test is only reported.</InfoTip>
        </CardTitle>
        {(every.length > rows.length || all) && (
          <CardAction>
            <Button variant="outline" size="sm" onClick={() => setAll(!all)}>
              <ListFilter />
              {all ? "Show the key models" : `Show all ${every.length}`}
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="space-y-6">
        {rows.length ? (
          <BoardChart rows={rows.map((r) => ({ ...r, model: display(d, r.model) }))} />
        ) : (
          <p className="text-muted-foreground text-sm">Nothing scored yet. Build or score this specialist to fill the board.</p>
        )}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>
                <span className="inline-flex items-center gap-1">
                  Test [95% CI]
                  <InfoTip>
                    The range the test score falls in when the test rows are resampled 1,000 times. With a bound like {d.metric}, a resample that misses the
                    bound scores 0, so a lower end of 0 means the model sits at the edge of the bound.
                  </InfoTip>
                </span>
              </TableHead>
              <TableHead>Calib</TableHead>
              {caughtLabels.map((l) => (
                <TableHead key={l}>Catches {l}</TableHead>
              ))}
              <TableHead>Params</TableHead>
              <TableHead>CPU per input</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const res = d.board[r.model][d.primary_suite]
              return (
                <TableRow key={r.model}>
                  <TableCell>
                    <span className="flex items-center gap-2">
                      <ModelName model={d.models[r.model]} shown={display(d, r.model)} />
                      {r.model === d.promoted && <Badge>Promoted</Badge>}
                      {d.baselines.includes(r.model) && <Badge variant="outline">Rival</Badge>}
                    </span>
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {res.unreached != null ? (
                      <Badge
                        variant="outline"
                        className="text-muted-foreground font-normal"
                        title="Never reaches the required precision on test, so it scores 0"
                      >
                        precision {(res.unreached * 100).toFixed(1)}%
                      </Badge>
                    ) : (
                      <span className="font-medium">{score(r.test)}</span>
                    )}{" "}
                    {res.unreached == null && res.ci && (
                      <span className="text-muted-foreground/70">
                        [{score(res.ci.lo)}, {score(res.ci.hi)}]
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">{score(r.calib)}</TableCell>
                  {caughtLabels.map((l) => (
                    <TableCell key={l} className="text-foreground/70 tabular-nums">
                      {res.caught?.[l] != null ? `${((res.caught[l] as number) * 100).toFixed(1)}%` : "-"}
                    </TableCell>
                  ))}
                  <TableCell className="text-foreground/70 tabular-nums">{params(res.params)}</TableCell>
                  <TableCell className="text-foreground/70 tabular-nums">
                    {ms(res.latency_ms)}
                    {res.timed_busy && (
                      <Badge variant="outline" className="text-muted-foreground ml-2 font-normal" title={`Timed at ${res.latency_load} load per core`}>
                        busy
                      </Badge>
                    )}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
        <SizeChart d={d} />
      </CardContent>
    </Card>
  )
}
