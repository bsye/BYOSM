import { Check, ExternalLink, Plus, Search } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { openJob } from "@/job-sheet"

type Candidate = { repo: string; downloads: number; coverage: number; tag: string; added: boolean; baseline: string }

const scorable = (c: Candidate) => c.coverage > 0 || !c.baseline.startsWith("hf:")

const SHOWN = 8

export function HfRivals({ name }: { name: string }) {
  const [q, setQ] = useState("")
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
  const add = (c: Candidate) =>
    runJob(["baseline", "add", name, c.baseline, "--score"])
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
                <TableHead className="text-right">Downloads</TableHead>
                <TableHead className="text-right">Labels</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...items].sort((a, b) => Number(scorable(b)) - Number(scorable(a))).slice(0, SHOWN).map((c) => (
                <TableRow key={c.repo}>
                  <TableCell className="font-medium">
                    <a className="hover:underline" href={`https://huggingface.co/${c.repo}`} target="_blank" rel="noreferrer">
                      {c.repo}
                      <ExternalLink className="text-muted-foreground ml-1 inline size-3" />
                    </a>
                  </TableCell>
                  <TableCell className="text-foreground/70 text-right tabular-nums">{c.downloads.toLocaleString()}</TableCell>
                  <TableCell className="text-foreground/70 text-right tabular-nums">{c.baseline.startsWith("hf:") ? `${Math.round(c.coverage * 100)}%` : "adapter"}</TableCell>
                  <TableCell className="text-right">
                    {c.added ? (
                      <Badge variant="outline">
                        <Check />
                        Rival
                      </Badge>
                    ) : (
                      <Button size="sm" variant="outline" disabled={!scorable(c)} onClick={() => add(c)}>
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
      </CardContent>
    </Card>
  )
}
