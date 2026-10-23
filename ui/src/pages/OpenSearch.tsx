import { Bookmark, Download, Eye, Search } from "lucide-react"
import { useState } from "react"
import { api } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { DataPreview } from "./DataPreview"
import { download, keep, useLibrary, type OpenDataset } from "./opendata-api"

const ALL = "all"
const SOURCES = ["data.gov.ie", "data.gov.uk", "data.europa.eu"]
const SKELETONS = 4

export function OpenSearch() {
  const [q, setQ] = useState("")
  const [source, setSource] = useState(ALL)
  const [found, setFound] = useState<OpenDataset[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [shown, setShown] = useState<{ url: string; title: string } | null>(null)
  const [kept, setKept] = useLibrary()
  const run = () => {
    setBusy(true)
    const filter = source === ALL ? "" : `&source=${encodeURIComponent(source)}`
    api<OpenDataset[]>(`/opendata/search?q=${encodeURIComponent(q)}${filter}`)
      .then(setFound, () => setFound([]))
      .finally(() => setBusy(false))
  }
  const cached = new Set((kept ?? []).filter((k) => k.cached).map((k) => k.file.url))
  const saved = new Set((kept ?? []).map((k) => k.file.url))
  return (
    <div className="space-y-4">
      <div className="mx-auto flex w-full max-w-3xl gap-2">
        <InputGroup>
          <InputGroupInput value={q} placeholder="companies register, business closures, inspections" onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && q && run()} />
          <InputGroupAddon align="inline-end">
            <InputGroupButton disabled={!q || busy} onClick={run}>
              <Search />
              Search
            </InputGroupButton>
          </InputGroupAddon>
        </InputGroup>
        <Select value={source} onValueChange={setSource}>
          <SelectTrigger className="w-44" aria-label="Portal">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All portals</SelectItem>
            {SOURCES.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {busy && Array.from({ length: SKELETONS }, (_, i) => <Skeleton key={i} className="h-24 rounded-xl" />)}
      {!busy && found?.length === 0 && <p className="text-muted-foreground text-center text-sm">No table found for that search.</p>}
      {!busy &&
        found?.map((d) => (
          <Card key={`${d.source}/${d.id}`}>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2">
                <a className="hover:underline" href={d.url} target="_blank" rel="noreferrer">
                  {d.title}
                </a>
                <Badge variant="outline">{d.source}</Badge>
              </CardTitle>
              <CardDescription className="line-clamp-2">
                {[d.publisher, d.licence, d.modified].filter(Boolean).join(" · ")}
                {d.notes ? ` · ${d.notes}` : ""}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2">
              {d.files.map((f) => (
                <div key={f.url} className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="secondary">{f.format}</Badge>
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
            </CardContent>
          </Card>
        ))}
      <DataPreview url={shown?.url ?? null} title={shown?.title ?? ""} onClose={() => setShown(null)} />
    </div>
  )
}
