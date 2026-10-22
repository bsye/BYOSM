import { Check, ExternalLink, ListFilter, Plus, Search } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { openJob } from "@/job-sheet"
import { MapLabels } from "./MapLabels"

type Candidate = { repo: string; downloads: number; coverage: number; tag: string; added: boolean; baseline: string; labels?: string[] }

const scorable = (c: Candidate) => c.coverage > 0 || !c.baseline.startsWith("hf:")
const mappable = (c: Candidate) => !scorable(c) && (c.labels?.length ?? 0) >= 2

const SHOWN = 8

const usable = (c: Candidate) => scorable(c) || mappable(c)
const hidden = (items: Candidate[]) => items.filter((c) => !usable(c)).length

function shown(items: Candidate[], all: boolean): Candidate[] {
  const ranked = [...items].sort((a, b) => Number(scorable(b)) * 2 + Number(mappable(b)) - Number(scorable(a)) * 2 - Number(mappable(a)))
  return all ? ranked : ranked.filter(usable).slice(0, SHOWN)
}

export function HfRivals({ name, ours }: { name: string; ours: string[] }) {
  const [q, setQ] = useState("")
  const [all, setAll] = useState(false)
  const [items, setItems] = useState<Candidate[] | null>(null)
  const [error, setError] = useState("")
  const search = (query: string) => {
    setItems(null)
    api<Candidate[]>(`/specialists/${name}/rivals/search?q=${encodeURIComponent(query)}`)
      .then(setItems)
      .catch((e) => setError(String(e)))
  }
  useEffect(() => {
    api<Candidate[]>(`/specialists/${name}/rivals/search`)
      .then(setItems)
      .catch((e) => setError(String(e)))
  }, [name])
  const add = (id: string) =>
    runJob(["baseline", "add", name, id, "--score"])
      .then((j) => openJob(j.id))
      .catch((e) => setError(String(e)))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Test against a Hugging Face model
          <InfoTip>Hub models for the same task, by downloads. Labels is the share of our labels the model can output; only those can be scored.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            search(q)
          }}
        >
          <Input className="h-8" placeholder="Search the Hub, e.g. emotion roberta" value={q} onChange={(e) => setQ(e.target.value)} />
          <Button size="sm" variant="outline" type="submit">
            <Search />
            Search
          </Button>
        </form>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {items == null && !error && <p className="text-muted-foreground text-sm">Searching the Hub.</p>}
        {items?.length === 0 && <p className="text-muted-foreground text-sm">No model for this task matches.</p>}
        {items && items.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead className="text-right">Downloads</TableHead>
                <TableHead className="text-right">Labels</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown(items, all).map((c) => (
                <TableRow key={c.repo}>
                  <TableCell className="font-medium">
                    <a className="hover:underline" href={`https://huggingface.co/${c.repo}`} target="_blank" rel="noreferrer">
                      {c.repo}
                      <ExternalLink className="text-muted-foreground ml-1 inline size-3" />
                    </a>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-xs">{c.tag || "-"}</TableCell>
                  <TableCell className="text-foreground/70 text-right tabular-nums">{c.downloads.toLocaleString()}</TableCell>
                  <TableCell className="text-foreground/70 text-right tabular-nums">{c.baseline.startsWith("hf:") ? `${Math.round(c.coverage * 100)}%` : "adapter"}</TableCell>
                  <TableCell className="text-right">
                    {c.added ? (
                      <Badge variant="outline">
                        <Check />
                        Rival
                      </Badge>
                    ) : mappable(c) ? (
                      <MapLabels repo={c.repo} theirs={c.labels ?? []} ours={ours} onAdd={add} />
                    ) : (
                      <Button size="sm" variant="outline" disabled={!scorable(c)} onClick={() => add(c.baseline)}>
                        <Plus />
                        Add and score
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        {items && hidden(items) > 0 && (
          <Button size="sm" variant="ghost" onClick={() => setAll(!all)}>
            <ListFilter />
            {all ? "Hide the ones that cannot be scored" : `Show ${hidden(items)} that cannot be scored`}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
