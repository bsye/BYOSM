import { ExternalLink, Play, Search } from "lucide-react"
import { runJob, type Detail } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { openJob } from "@/job-sheet"
import { modelName } from "@/jobs"

export type DataCheck = {
  verdict: "too small" | "thin" | "enough"
  train: number
  labels: Record<string, { rows: number; min: number; recommended: number }>
  short: string[]
  thin: string[]
  test: number
  calib: number
  min_test: number
  min_calib: number
}
export type DataSuggestion = { dataset: string; role: string; why: string; by: string; args: string[] }

const SHOWN_LABELS = 8
const ANSWER = { "too small": "Too small", thin: "Thin", enough: "Enough" }

const start = (args: string[]) => runJob(args).then((j) => openJob(j.id))

function Figure({ label, value, need }: { label: string; value: number; need?: number }) {
  const low = need != null && value < need
  return (
    <div>
      <p className="text-muted-foreground text-xs">{label}</p>
      <p className={`text-lg font-semibold tabular-nums ${low ? "text-destructive" : ""}`}>
        {value.toLocaleString()}
        {low && <span className="text-muted-foreground ml-1 text-xs font-normal">needs {need}+</span>}
      </p>
    </div>
  )
}

function ThinLabels({ c }: { c: DataCheck }) {
  const rows = [...c.thin]
    .sort((a, b) => c.labels[a].rows - c.labels[b].rows)
    .slice(0, SHOWN_LABELS)
    .map((n) => [n, c.labels[n]] as const)
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Label below the recommended rows</TableHead>
          <TableHead className="text-right">Rows</TableHead>
          <TableHead className="text-right">Recommended</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(([n, x]) => (
          <TableRow key={n}>
            <TableCell className="font-medium">{n}</TableCell>
            <TableCell className={`text-right tabular-nums ${x.rows < x.min ? "text-destructive" : ""}`}>{x.rows}</TableCell>
            <TableCell className="text-foreground/70 text-right tabular-nums">{x.recommended}</TableCell>
          </TableRow>
        ))}
        {c.thin.length > SHOWN_LABELS && (
          <TableRow>
            <TableCell colSpan={3} className="text-muted-foreground text-xs">
              and {c.thin.length - SHOWN_LABELS} more
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  )
}

function Suggestions({ items }: { items: DataSuggestion[] }) {
  if (!items.length) return <p className="text-muted-foreground text-sm">The last search found no dataset whose columns map onto these labels.</p>
  return (
    <ul className="divide-y border-y">
      {items.map((s) => (
        <li key={s.dataset} className="flex items-start gap-3 py-2.5">
          <div className="min-w-0 flex-1">
            <a className="font-medium hover:underline" href={`https://huggingface.co/datasets/${s.dataset}`} target="_blank" rel="noreferrer">
              {s.dataset}
              <ExternalLink className="text-muted-foreground ml-1 inline size-3" />
            </a>
            <p className="text-muted-foreground text-xs">{s.why}</p>
          </div>
          <Badge variant="outline" className="text-muted-foreground font-normal">
            {modelName(s.by)}
          </Badge>
          <Button size="sm" variant="outline" onClick={() => start(s.args)}>
            <Play />
            Train with it
          </Button>
        </li>
      ))}
    </ul>
  )
}

export function DataSize({ d }: { d: Detail }) {
  const c = d.data_check as DataCheck | null
  if (!c) return null
  const found = d.data_suggestions as DataSuggestion[] | null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Training data</CardTitle>
        <CardAction>
          <Button variant="outline" size="sm" onClick={() => start(["datasets", d.name])}>
            <Search />
            {found ? "Search again" : "Find datasets"}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-semibold">{ANSWER[c.verdict]}</span>
          <span className="text-muted-foreground text-sm">for a model that generalises</span>
        </div>
        <div className="grid grid-cols-2 gap-4 @3xl/main:grid-cols-4">
          <Figure label="Training rows" value={c.train} />
          <Figure label="Labels below recommended" value={c.thin.length} />
          <Figure label="Test rows" value={c.test} need={c.min_test} />
          <Figure label="Calib rows" value={c.calib} need={c.min_calib} />
        </div>
        {c.thin.length > 0 && <ThinLabels c={c} />}
        {found && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Datasets that fit</h3>
            <Suggestions items={found} />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
