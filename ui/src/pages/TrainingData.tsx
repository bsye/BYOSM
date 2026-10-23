import { useEffect, useState } from "react"
import { api } from "@/api"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { href } from "@/route"
import { DataRows } from "./DataRows"

type Built = { name: string; title: string; modality: string; labels: number; splits: Record<string, number>; sources: string[] }
const SPLITS = ["train", "dev", "test"]

export function TrainingData() {
  const [rows, setRows] = useState<Built[] | null>(null)
  const [picked, setPicked] = useState<string | null>(null)
  useEffect(() => {
    api<Built[]>("/datasets/training").then(setRows, () => setRows([]))
  }, [])
  const { shown, footer } = useShowMore(rows ?? [])
  if (!rows) return <Skeleton className="h-64 rounded-xl" />
  const chosen = picked ?? rows[0]?.name
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Built for your specialists</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Specialist</TableHead>
                <TableHead>Modality</TableHead>
                <TableHead className="text-right">Labels</TableHead>
                {SPLITS.map((s) => (
                  <TableHead key={s} className="text-right">
                    {s}
                  </TableHead>
                ))}
                <TableHead>Sources</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((r) => (
                <TableRow key={r.name} className="cursor-pointer" data-state={chosen === r.name ? "selected" : undefined} onClick={() => setPicked(r.name)}>
                  <TableCell className="font-medium">
                    <a className="hover:underline" href={href("specialists", r.name)} onClick={(e) => e.stopPropagation()}>
                      {r.title}
                    </a>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{r.modality}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-right tabular-nums">{r.labels}</TableCell>
                  {SPLITS.map((s) => (
                    <TableCell key={s} className="text-right tabular-nums">
                      {(r.splits[s] ?? 0).toLocaleString()}
                    </TableCell>
                  ))}
                  <TableCell className="text-muted-foreground max-w-56 truncate text-xs">{r.sources.join(", ") || "written by an LLM"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
        {footer}
      </Card>
      {chosen && <DataRows key={chosen} name={chosen} />}
    </div>
  )
}
