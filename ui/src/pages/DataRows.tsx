import { useMemo, useState } from "react"
import { api } from "@/api"
import { LoadMore, useInfinite } from "@/components/infinite"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

const SPLITS = ["train", "dev", "calib", "test"]

type Row = { id: string; input: unknown; label: unknown; source: string }

export function DataRows({ name }: { name: string }) {
  const [split, setSplit] = useState("train")
  const fetchPage = useMemo(
    () => (offset: number, limit: number) =>
      api<{ total: number; rows: Row[] }>(`/specialists/${name}/rows?split=${split}&offset=${offset}&limit=${limit}`).then((p) => ({ ...p, extra: null })),
    [name, split],
  )
  const page = useInfinite(`${name}/${split}`, fetchPage, () => null)
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Rows <span className="text-muted-foreground font-normal tabular-nums">{page.total.toLocaleString()}</span>
        </CardTitle>
        <CardAction>
          <Select value={split} onValueChange={setSplit}>
            <SelectTrigger size="sm" className="w-32" aria-label="Split">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SPLITS.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardAction>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[32rem]">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Input</TableHead>
                <TableHead>Label</TableHead>
                <TableHead>Source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {page.rows.map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="max-w-xl truncate">{String(r.input)}</TableCell>
                  <TableCell className="text-foreground/70 max-w-xs truncate">{typeof r.label === "string" ? r.label : JSON.stringify(r.label)}</TableCell>
                  <TableCell className="text-muted-foreground text-xs">{r.source}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <LoadMore onVisible={page.more} loading={page.loading} done={page.done} shown={page.rows.length} total={page.total} />
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
