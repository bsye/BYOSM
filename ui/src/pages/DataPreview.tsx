import { useEffect, useState } from "react"
import { api } from "@/api"
import { Badge } from "@/components/ui/badge"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Preview } from "./opendata-api"

export const filePreview = (url: string) => `/opendata/preview?url=${encodeURIComponent(url)}`

export function DataPreview({ path, title, onClose }: { path: string | null; title: string; onClose: () => void }) {
  const [data, setData] = useState<Preview | null>(null)
  const [error, setError] = useState("")
  useEffect(() => {
    setData(null)
    setError("")
    if (path) api<Preview>(path).then(setData, (e) => setError(String(e)))
  }, [path])
  return (
    <Sheet open={path != null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="w-full overflow-y-auto data-[side=right]:sm:max-w-4xl">
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
          <SheetDescription>
            {!data && "Reading the file"}
            {data && data.match != null && `${Math.round(data.match * 100)}% of the first ${data.sampled.toLocaleString()} left rows found a match`}
            {data && data.match == null && `${data.file}, first ${data.sampled.toLocaleString()} rows profiled`}
          </SheetDescription>
        </SheetHeader>
        <div className="space-y-6 px-4 pb-6">
          {error && <p className="text-destructive text-sm">{error}</p>}
          {!data && !error && <Skeleton className="h-64" />}
          {data && (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Column</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead className="text-right">Missing</TableHead>
                    <TableHead className="text-right">Distinct</TableHead>
                    <TableHead>Examples</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.columns.map((c) => (
                    <TableRow key={c.name}>
                      <TableCell className="font-medium">{c.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{c.type}</Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground text-right tabular-nums">{Math.round(c.missing * 100)}%</TableCell>
                      <TableCell className="text-muted-foreground text-right tabular-nums">{c.distinct.toLocaleString()}</TableCell>
                      <TableCell className="text-muted-foreground max-w-72 truncate text-xs">{c.examples.join(", ")}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <ScrollArea className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {data.columns.map((c) => (
                        <TableHead key={c.name}>{c.name}</TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.rows.map((r, i) => (
                      <TableRow key={i}>
                        {data.columns.map((c) => (
                          <TableCell key={c.name} className="max-w-48 truncate text-xs">
                            {r[c.name] ?? ""}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <ScrollBar orientation="horizontal" />
              </ScrollArea>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}
