import { RefreshCw } from "lucide-react"
import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, XAxis, YAxis } from "recharts"
import { api, put, runJob, type Detail, type PanelView, type ReviewRow } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { score } from "@/format"
import { go } from "@/route"

const PANEL = "panel"
const config = { accuracy: { label: "accuracy on labelled rows", color: "var(--chart-1)" } } satisfies ChartConfig
const percent = (x: number) => `${Math.round(x * 100)}%`

export const hasTeachers = (d: Detail) => Boolean((d.spec.data as { teachers?: string[] } | undefined)?.teachers?.length)

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="text-2xl font-semibold tabular-nums">{value}</div>
    </div>
  )
}

function Accuracy({ view }: { view: PanelView }) {
  const s = view.summary
  if (!s) {
    return (
      <Alert>
        <AlertTitle>No panel yet</AlertTitle>
        <AlertDescription>Build the data once and the teachers label the unlabeled inputs.</AlertDescription>
      </Alert>
    )
  }
  const rows = [
    ...Object.entries(s.teachers).map(([model, accuracy]) => ({ model, accuracy })),
    { model: PANEL, accuracy: s.panel_on_labelled },
  ]
  return (
    <Card>
      <CardHeader>
        <CardTitle>Teachers</CardTitle>
        <CardDescription>
          Accuracy of each teacher and of the weighted panel on the {s.n_labelled} labelled rows. The panel labelled{" "}
          {s.n_pool} inputs for training.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-[1fr_14rem]">
        <ChartContainer config={config} className="h-64 w-full">
          <BarChart data={rows} margin={{ top: 20, left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="model" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} tickLine={false} axisLine={false} width={40} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="accuracy" radius={4}>
              <LabelList dataKey="accuracy" position="top" formatter={(v) => score(Number(v))} className="fill-foreground" />
              {rows.map((r) => (
                <Cell key={r.model} fill={r.model === PANEL ? "var(--chart-2)" : "var(--color-accuracy)"} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
        <div className="grid content-start gap-3">
          <Stat label="all teachers agree" value={percent(s.all_agree)} />
          <Stat label="reviewed by a human" value={String(view.n_reviewed)} />
        </div>
      </CardContent>
    </Card>
  )
}

function ReviewLine({ row, labels, onSave }: { row: ReviewRow; labels: string[]; onSave: (label: string) => void }) {
  return (
    <TableRow>
      <TableCell className="max-w-md whitespace-normal">{row.input}</TableCell>
      <TableCell className="font-mono text-xs">
        {row.panel} <span className="text-muted-foreground">{percent(row.sure)}</span>
      </TableCell>
      <TableCell className="space-x-1 space-y-1 whitespace-normal">
        {Object.entries(row.votes).map(([teacher, label]) => (
          <Badge key={teacher} variant={label === row.panel ? "secondary" : "outline"} className="font-mono">
            {teacher}: {label ?? "-"}
          </Badge>
        ))}
      </TableCell>
      <TableCell className="space-y-1">
        <Select value={row.reviewed ?? undefined} onValueChange={onSave}>
          <SelectTrigger className="w-56">
            <SelectValue placeholder="set the label" />
          </SelectTrigger>
          <SelectContent>
            {labels.map((l) => (
              <SelectItem key={l} value={l}>
                {l}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {row.reviewed && <Badge>reviewed</Badge>}
      </TableCell>
    </TableRow>
  )
}

function Review({ name, view, onSaved }: { name: string; view: PanelView; onSaved: (row: ReviewRow) => void }) {
  const [error, setError] = useState("")
  const save = (row: ReviewRow, label: string) =>
    put(`/specialists/${name}/panel/review`, { input: row.input, label })
      .then(() => onSaved({ ...row, reviewed: label }))
      .catch((e) => setError(String(e)))
  const rebuild = () => runJob(["data", name, "--force"]).then((j) => go("jobs", j.id))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Least sure rows</CardTitle>
        <CardDescription>
          The {view.review.length} inputs the panel is least sure about. Set the right label and it replaces the panel's guess.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert>
          <AlertTitle>Reviews take effect on the next data rebuild</AlertTitle>
          <AlertDescription className="flex flex-wrap items-center gap-3">
            <span>
              Run <code>byosm data {name} --force</code>, or an experiment with a new data key, to train on them.
            </span>
            <Button size="sm" variant="outline" onClick={rebuild}>
              <RefreshCw />
              Rebuild the data
            </Button>
          </AlertDescription>
        </Alert>
        {error && <p className="text-destructive text-sm">{error}</p>}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>input</TableHead>
              <TableHead>panel</TableHead>
              <TableHead>teachers</TableHead>
              <TableHead>human label</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {view.review.map((row) => (
              <ReviewLine key={row.input} row={row} labels={view.labels} onSave={(label) => save(row, label)} />
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export function Panel({ name }: { name: string }) {
  const [view, setView] = useState<PanelView | null>(null)
  const [error, setError] = useState("")
  useEffect(() => {
    api<PanelView>(`/specialists/${name}/panel`).then(setView).catch((e) => setError(String(e)))
  }, [name])
  if (error) return <p className="text-destructive">{error}</p>
  if (!view) return null
  const saved = (row: ReviewRow) => {
    const fresh = !view.review.find((r) => r.input === row.input)?.reviewed
    setView({
      ...view,
      n_reviewed: view.n_reviewed + (fresh ? 1 : 0),
      review: view.review.map((r) => (r.input === row.input ? row : r)),
    })
  }
  return (
    <div className="space-y-4">
      <Accuracy view={view} />
      {view.review.length > 0 && <Review name={name} view={view} onSaved={saved} />}
    </div>
  )
}
