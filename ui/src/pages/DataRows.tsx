import { ChevronLeft, ChevronRight } from "lucide-react"
import { useEffect, useState } from "react"
import { api } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

const PAGE = 50
const SPLITS = ["train", "dev", "test"]

type Row = { id: string; input: unknown; label: unknown; source: string }

export function DataRows({ name }: { name: string }) {
  const [split, setSplit] = useState("train")
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<{ total: number; rows: Row[] }>({ total: 0, rows: [] })
  useEffect(() => {
    api<{ total: number; rows: Row[] }>(`/specialists/${name}/rows?split=${split}&offset=${offset}&limit=${PAGE}`)
      .then(setPage)
      .catch(() => setPage({ total: 0, rows: [] }))
  }, [name, split, offset])
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-4">
        <div>
          <CardTitle>Data</CardTitle>
          <CardDescription>
            {page.total} rows in {split}
          </CardDescription>
        </div>
        <Select
          value={split}
          onValueChange={(v) => {
            setSplit(v)
            setOffset(0)
          }}
        >
          <SelectTrigger className="w-32">
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
      </CardHeader>
      <CardContent className="space-y-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>input</TableHead>
              <TableHead>label</TableHead>
              <TableHead>source</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {page.rows.map((r) => (
              <TableRow key={r.id}>
                <TableCell className="max-w-xl truncate">{String(r.input)}</TableCell>
                <TableCell className="max-w-xs truncate font-mono text-xs">{JSON.stringify(r.label)}</TableCell>
                <TableCell className="text-muted-foreground text-xs">{r.source}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            <ChevronLeft />
            Previous
          </Button>
          <Button variant="outline" size="sm" disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)}>
            Next
            <ChevronRight />
          </Button>
          <span className="text-muted-foreground text-xs">
            {page.total ? `${offset + 1} to ${Math.min(offset + PAGE, page.total)}` : ""}
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
