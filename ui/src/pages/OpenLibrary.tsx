import { Download, Eye, Trash2 } from "lucide-react"
import { useState } from "react"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { DataPreview, filePreview } from "./DataPreview"
import { JoinBuilder } from "./JoinBuilder"
import { download, forget, useLibrary } from "./opendata-api"

export function OpenLibrary() {
  const [kept, setKept] = useLibrary()
  const [shown, setShown] = useState<{ url: string; title: string } | null>(null)
  const { shown: rows, footer } = useShowMore(kept ?? [])
  if (kept == null) return <Skeleton className="h-48 rounded-xl" />
  if (!kept.length)
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>Nothing kept yet</EmptyTitle>
          <EmptyDescription>Find data and keep the tables worth training on.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Kept tables</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Licence</TableHead>
                <TableHead>File</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((k) => (
                <TableRow key={k.file.url}>
                  <TableCell className="max-w-64 truncate font-medium">
                    <a className="hover:underline" href={k.url} target="_blank" rel="noreferrer">
                      {k.title}
                    </a>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{k.source}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground max-w-40 truncate">{k.licence}</TableCell>
                  <TableCell className="text-muted-foreground">
                    <Badge variant={k.cached ? "secondary" : "outline"}>{k.cached ? "Downloaded" : k.file.format}</Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      {k.cached ? (
                        <Button size="sm" variant="outline" onClick={() => setShown({ url: k.file.url, title: k.title })}>
                          <Eye />
                          Preview
                        </Button>
                      ) : (
                        <Button size="sm" variant="outline" onClick={() => void download(k.file.url)}>
                          <Download />
                          Download
                        </Button>
                      )}
                      <Button size="sm" variant="ghost" aria-label="Remove from the library" onClick={() => void forget(k.file.url).then(setKept)}>
                        <Trash2 />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
        {footer}
        <DataPreview path={shown ? filePreview(shown.url) : null} title={shown?.title ?? ""} onClose={() => setShown(null)} />
      </Card>
      <JoinBuilder kept={kept} />
    </div>
  )
}
