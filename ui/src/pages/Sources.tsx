import { Info, Plus, Trash2 } from "lucide-react"
import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { addSource, dropSource, type Source } from "./opendata-api"

const KIND = { ckan: "Portal", socrata: "Portal", page: "Page" }
const BUILT_IN = "built in"
const BY = { you: "You", found: "LLM", [BUILT_IN]: "Built in" } as Record<string, string>

function AddSource({ onAdded }: { onAdded: (s: Source[]) => void }) {
  const [url, setUrl] = useState("")
  const [name, setName] = useState("")
  const [busy, setBusy] = useState(false)
  const add = () => {
    setBusy(true)
    addSource(url, name)
      .then(
        (s) => {
          onAdded(s)
          setUrl("")
          setName("")
        },
        (e) => toast.error(String(e)),
      )
      .finally(() => setBusy(false))
  }
  return (
    <div className="flex flex-col gap-2 md:flex-row">
      <Input value={url} placeholder="https://data.example.org" onChange={(e) => setUrl(e.target.value)} onKeyDown={(e) => e.key === "Enter" && url && add()} />
      <Input className="md:w-56" value={name} placeholder="Name" onChange={(e) => setName(e.target.value)} />
      <Button disabled={!url.trim() || busy} onClick={add}>
        <Plus />
        {busy ? "Checking" : "Add"}
      </Button>
    </div>
  )
}

export function Sources() {
  const [rows, setRows] = useState<Source[] | null>(null)
  useEffect(() => {
    api<Source[]>("/opendata/sources").then(setRows, () => setRows([]))
  }, [])
  const { shown, footer } = useShowMore(rows ?? [])
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Sources
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="text-muted-foreground size-4" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              Portals are searched on every request. Pages are given to the LLM as places to look. Searches also add portals they discover.
            </TooltipContent>
          </Tooltip>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <AddSource onAdded={setRows} />
        {!rows ? (
          <Skeleton className="h-40 rounded-lg" />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Added by</TableHead>
                <TableHead className="text-right">Remove</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((s) => (
                <TableRow key={s.url}>
                  <TableCell className="max-w-80">
                    <a className="block truncate font-medium hover:underline" href={s.url} target="_blank" rel="noreferrer" title={s.why || s.url}>
                      {s.name}
                    </a>
                    <div className="text-muted-foreground truncate text-xs">{s.why || s.url}</div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={s.kind !== "page" ? "secondary" : "outline"}>{KIND[s.kind]}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{BY[s.by] ?? s.by}</TableCell>
                  <TableCell className="text-right">
                    <Button
                      size="icon"
                      variant="ghost"
                      disabled={s.by === BUILT_IN}
                      aria-label="Remove source"
                      onClick={() => void dropSource(s.url).then(setRows)}
                    >
                      <Trash2 />
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
