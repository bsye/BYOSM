import { Copy } from "lucide-react"
import { useEffect, useState } from "react"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { useShowMore } from "@/components/show-more"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { compact, params } from "@/format"

const SKELETON_ROWS = 6

type Trending = { id: string; task: string; downloads: number; likes: number; params: number | null }

const TASK_WORDS: Record<string, string> = {
  "text-classification": "classifies text",
  "token-classification": "tags words in text",
  "audio-classification": "classifies audio",
  "image-classification": "classifies images",
}

export function replicatePrompt(t: Trending): string {
  return `do what the Hugging Face model ${t.id} does (it ${TASK_WORDS[t.task] ?? t.task}), smaller and at least as accurate`
}

export function Opportunities({ onPick }: { onPick: (prompt: string) => void }) {
  const [rows, setRows] = useState<Trending[] | null>(null)
  useEffect(() => {
    api<Trending[]>("/opportunities").then(setRows, () => setRows([]))
  }, [])
  const { shown, footer } = useShowMore(rows ?? [])
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Trending on Hugging Face
          <InfoTip>Trending models in the tasks BYOSM builds. Replicate one with a small specialist and score it as a rival.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {rows == null ? (
          <div className="space-y-3">
            {Array.from({ length: SKELETON_ROWS }, (_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Downloads</TableHead>
                <TableHead className="text-right">Start</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((t) => (
                <TableRow key={t.id}>
                  <TableCell>
                    <a
                      className="block max-w-56 truncate font-medium hover:underline"
                      href={`https://huggingface.co/${t.id}`}
                      title={`${t.id}, ${t.task}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {t.id}
                    </a>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-right tabular-nums">{t.params ? params(t.params) : "-"}</TableCell>
                  <TableCell className="text-muted-foreground text-right tabular-nums">{compact(t.downloads)}</TableCell>
                  <TableCell className="text-right">
                    <Button size="sm" variant="outline" onClick={() => onPick(replicatePrompt(t))}>
                      <Copy />
                      Replicate
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
      {footer}
    </Card>
  )
}
