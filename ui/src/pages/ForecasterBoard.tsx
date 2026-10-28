import { Loader2 } from "lucide-react"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, XAxis, YAxis } from "recharts"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ms, score } from "@/format"
import { isRival, trainTime, type ForecasterView, type Score } from "./forecaster-api"

const chart = {
  mase: { label: "Test MASE", color: "var(--chart-1)" },
} satisfies ChartConfig

const OURS_FILL = "var(--chart-1)"
const RIVAL_FILL = "var(--muted-foreground)"

function Who({ name, f, s }: { name: string; f: ForecasterView; s: Score }) {
  if (isRival(name)) return <Badge variant="outline">{s.published ? "published rival" : "rival"}</Badge>
  const promoted = f.versions?.find((v) => v.name === f.promoted)?.recipe === name
  return promoted ? <Badge>ours, {f.promoted}</Badge> : <Badge variant="secondary">ours</Badge>
}

function Coverage({ s, total }: { s: Score; total: number }) {
  const n = s.datasets ?? total
  if (n >= total) return <>{`${n} of ${total}`}</>
  return (
    <span className="inline-flex items-center gap-1">
      {`${n} of ${total}`}
      {s.ours_on_shared != null && <InfoTip>{`On the same ${n} datasets our promoted model scores ${score(s.ours_on_shared)}.`}</InfoTip>}
    </span>
  )
}

export function Board({ f, running }: { f: ForecasterView; running: boolean }) {
  const models = Object.entries(f.overall ?? {}).sort(([a, x], [b, y]) => Number(isRival(a)) - Number(isRival(b)) || x.mase - y.mase)
  const total = Object.keys(f.configs ?? {}).length
  return (
    <Card>
      <CardHeader>
        <CardTitle>Against the rivals</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <ChartContainer config={chart} className="h-56 w-full">
          <BarChart data={models.map(([name, s]) => ({ name, mase: s.mase }))} margin={{ top: 20, left: 8, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} fontSize={11} />
            <YAxis tickLine={false} axisLine={false} width={36} tickFormatter={(v) => Number(v).toFixed(2)} />
            <ReferenceLine y={1} strokeDasharray="4 4" />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="mase" radius={4}>
              {models.map(([name]) => (
                <Cell key={name} fill={isRival(name) ? RIVAL_FILL : OURS_FILL} />
              ))}
              <LabelList dataKey="mase" position="top" fontSize={11} className="fill-foreground" formatter={(v) => score(Number(v))} />
            </Bar>
          </BarChart>
        </ChartContainer>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead />
              <TableHead className="text-right">Datasets</TableHead>
              <TableHead className="text-right">Calib MASE</TableHead>
              <TableHead className="text-right">Test MASE</TableHead>
              <TableHead className="text-right">Test CRPS</TableHead>
              <TableHead className="text-right">Train</TableHead>
              <TableHead className="text-right">Per forecast</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {models.map(([name, s]) => (
              <TableRow key={name}>
                <TableCell className="font-medium">{name}</TableCell>
                <TableCell>
                  <Who name={name} f={f} s={s} />
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  <Coverage s={s} total={total} />
                </TableCell>
                <TableCell className="text-right tabular-nums">{score(s.calib_mase)}</TableCell>
                <TableCell className="text-right tabular-nums">{score(s.mase)}</TableCell>
                <TableCell className="text-right tabular-nums">{score(s.crps)}</TableCell>
                <TableCell className="text-right tabular-nums">{trainTime(s.train_s)}</TableCell>
                <TableCell className="text-right tabular-nums">{ms(s.ms_per_forecast)}</TableCell>
              </TableRow>
            ))}
            {(running ? (f.pending ?? []) : []).map((name) => (
              <TableRow key={name}>
                <TableCell className="text-muted-foreground flex items-center gap-2">
                  <Loader2 className="size-3.5 animate-spin" />
                  {name}
                </TableCell>
                <TableCell colSpan={7} className="text-muted-foreground text-right">
                  training
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
