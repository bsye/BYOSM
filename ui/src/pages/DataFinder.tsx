import { Loader2, Search, Sparkles, Trash2 } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"
import { api } from "@/api"
import { useShowMore } from "@/components/show-more"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { InputGroup, InputGroupAddon, InputGroupTextarea } from "@/components/ui/input-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { FoundDatasets } from "./FoundDatasets"
import { ask, forgetAsk, useAsks, type DataAsk, type OpenDataset } from "./opendata-api"

const ASK = "ask"
const SEARCH = "search"
const EXAMPLES = ["Irish companies that closed, with their sector and county", "Food safety inspections of restaurants with their outcome", "Planning applications and whether they were granted"]
const SKELETONS = 3
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

function AskCard({ item, onForget }: { item: DataAsk; onForget: () => void }) {
  const running = item.status === "running"
  const { shown, footer } = useShowMore(item.datasets)
  const stage = running ? (item.queries.length ? "Searching portals" : "Searching the web") : ""
  const facts = [stage, plural(item.datasets.length, "table"), item.portals.length ? plural(item.portals.length, "new portal") : "", item.queries.length ? `searched ${item.queries.length}` : ""]
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle className="line-clamp-2">{item.request}</CardTitle>
        <CardDescription className="flex items-center gap-2">
          {running && <Loader2 className="size-3.5 animate-spin" />}
          {item.status === "failed" && <Badge variant="destructive">Failed</Badge>}
          {facts.filter(Boolean).join(" · ")}
        </CardDescription>
        <CardAction>
          <Button size="icon" variant="ghost" aria-label="Remove this search" onClick={onForget}>
            <Trash2 />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {item.datasets.length ? <FoundDatasets datasets={shown} /> : running ? <Skeleton className="h-16 rounded-lg" /> : <p className="text-muted-foreground text-sm">No downloadable table found.</p>}
      </CardContent>
      {footer}
    </Card>
  )
}

function Asks({ asks, onForget }: { asks: DataAsk[] | null; onForget: (a: DataAsk[]) => void }) {
  if (!asks) return <Skeleton className="h-40 rounded-xl" />
  if (!asks.length)
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No searches yet</EmptyTitle>
          <EmptyDescription>Describe the data you need above.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  return (
    <div className="grid gap-4">
      {asks.map((a) => (
        <AskCard key={a.request} item={a} onForget={() => void forgetAsk(a.request).then(onForget)} />
      ))}
    </div>
  )
}

function Results({ found }: { found: OpenDataset[] | null | "busy" }) {
  const { shown, footer } = useShowMore(Array.isArray(found) ? found : [])
  if (found === "busy") return <>{Array.from({ length: SKELETONS }, (_, i) => <Skeleton key={i} className="h-16 rounded-xl" />)}</>
  if (!found) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Portal results</CardTitle>
        <CardDescription>{plural(found.length, "table")}</CardDescription>
      </CardHeader>
      <CardContent>{found.length ? <FoundDatasets datasets={shown} /> : <p className="text-muted-foreground text-sm">No table found for that search.</p>}</CardContent>
      {footer}
    </Card>
  )
}

export function DataFinder() {
  const [text, setText] = useState("")
  const [mode, setMode] = useState(ASK)
  const [found, setFound] = useState<OpenDataset[] | null | "busy">(null)
  const [asks, setAsks] = useAsks()
  const go = () => {
    const q = text.trim()
    if (!q) return
    if (mode === SEARCH) {
      setFound("busy")
      api<OpenDataset[]>(`/opendata/search?q=${encodeURIComponent(q)}`).then(setFound, () => setFound([]))
      return
    }
    setFound(null)
    ask(q).then(
      () => setText(""),
      (e) => toast.error(String(e)),
    )
  }
  return (
    <div className="grid gap-4">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-3 py-2">
        <h2 className="text-center text-2xl font-semibold tracking-tight">What data do you need?</h2>
        <InputGroup>
          <InputGroupTextarea
            className="min-h-20"
            value={text}
            placeholder={mode === ASK ? EXAMPLES[0] : "companies register"}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                go()
              }
            }}
          />
          <InputGroupAddon align="block-end" className="justify-between">
            <Select value={mode} onValueChange={setMode}>
              <SelectTrigger size="sm" className="w-44" aria-label="How to find">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ASK}>
                  <Sparkles />
                  Ask the LLM
                </SelectItem>
                <SelectItem value={SEARCH}>
                  <Search />
                  Search portals
                </SelectItem>
              </SelectContent>
            </Select>
            <Button size="sm" disabled={!text.trim()} onClick={go}>
              {mode === ASK ? <Sparkles /> : <Search />}
              {mode === ASK ? "Find data" : "Search"}
            </Button>
          </InputGroupAddon>
        </InputGroup>
        {mode === ASK && !text && (
          <div className="flex flex-wrap justify-center gap-2">
            {EXAMPLES.slice(1).map((x) => (
              <Button key={x} size="sm" variant="ghost" className="text-muted-foreground" onClick={() => setText(x)}>
                {x}
              </Button>
            ))}
          </div>
        )}
      </div>
      <Results found={found} />
      <Asks asks={asks} onForget={setAsks} />
    </div>
  )
}
