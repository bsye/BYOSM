import type { Ablation } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { signed } from "@/format"

const VERDICT_VARIANT = { better: "default", worse: "destructive", "not sure": "outline" } as const

function shortValue(v: unknown): string {
  const text = String(v)
  return text.startsWith("hf:") ? text.split("/").at(-1) ?? text : text
}

function changeLabel(change: Record<string, unknown>): string {
  return Object.entries(change)
    .map(([k, v]) => `${k.split(".").at(-1)} = ${shortValue(v)}`)
    .join(", ")
}

export function Ablations({ items }: { items: Ablation[] }) {
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Ablations
          <InfoTip>
            Each change trained with and without it on matched seeds. Better or worse only when every seed agrees with a sure calib lead; test is
            only reported.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Change</TableHead>
              <TableHead>Calib change</TableHead>
              <TableHead>Per seed</TableHead>
              <TableHead>Test change</TableHead>
              <TableHead>Verdict</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((a) => (
              <TableRow key={changeLabel(a.change)}>
                <TableCell className="font-medium">{changeLabel(a.change)}</TableCell>
                <TableCell className="tabular-nums">{signed(a.calib)}</TableCell>
                <TableCell className="text-muted-foreground tabular-nums">{a.seeds.map((s) => signed(s.calib)).join("  ")}</TableCell>
                <TableCell className="text-foreground/70 tabular-nums">{signed(a.test)}</TableCell>
                <TableCell>
                  <Badge variant={VERDICT_VARIANT[a.verdict]}>{a.verdict}</Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
