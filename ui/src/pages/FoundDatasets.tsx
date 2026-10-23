import { Bookmark, Download, Eye } from "lucide-react"
import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { DataPreview, filePreview } from "./DataPreview"
import { download, keep, useLibrary, type OpenDataset } from "./opendata-api"

export function FoundDatasets({ datasets }: { datasets: OpenDataset[] }) {
  const [kept, setKept] = useLibrary()
  const [shown, setShown] = useState<{ url: string; title: string } | null>(null)
  const cached = new Set((kept ?? []).filter((k) => k.cached).map((k) => k.file.url))
  const saved = new Set((kept ?? []).map((k) => k.file.url))
  return (
    <div className="grid min-w-0 divide-y">
      {datasets.map((d) => (
        <div key={`${d.source}/${d.id}`} className="grid min-w-0 gap-2 py-3 first:pt-0 last:pb-0">
          <div className="flex min-w-0 items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <a className="truncate font-medium hover:underline" href={d.url || d.files[0]?.url} target="_blank" rel="noreferrer">
                  {d.title}
                </a>
              </TooltipTrigger>
              {(d.why || d.notes) && <TooltipContent className="max-w-sm">{d.why || d.notes}</TooltipContent>}
            </Tooltip>
            <Badge variant="outline" className="shrink-0">
              {d.source}
            </Badge>
            <span className="text-muted-foreground ml-auto hidden max-w-[40%] shrink-0 truncate text-xs md:block">{[d.publisher, d.licence].filter(Boolean).join(" · ")}</span>
          </div>
          {d.files.map((f) => (
            <div key={f.url} className="flex min-w-0 items-center gap-2 text-sm">
              <Badge variant="secondary" className="w-14 justify-center">
                {f.format || "FILE"}
              </Badge>
              <span className="text-muted-foreground min-w-0 flex-1 truncate">{f.name}</span>
              {cached.has(f.url) ? (
                <Button size="sm" variant="outline" onClick={() => setShown({ url: f.url, title: d.title })}>
                  <Eye />
                  Preview
                </Button>
              ) : (
                <Button size="sm" variant="outline" onClick={() => void keep(d, f).then(setKept).then(() => download(f.url))}>
                  <Download />
                  Download
                </Button>
              )}
              <Button size="sm" variant="ghost" disabled={saved.has(f.url)} onClick={() => void keep(d, f).then(setKept)}>
                <Bookmark />
                {saved.has(f.url) ? "Kept" : "Keep"}
              </Button>
            </div>
          ))}
        </div>
      ))}
      <DataPreview path={shown ? filePreview(shown.url) : null} title={shown?.title ?? ""} onClose={() => setShown(null)} />
    </div>
  )
}
