import { Loader2, Plus, Shuffle } from "lucide-react"
import { useState } from "react"
import { api } from "@/api"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const IGNORE = "__ignore__"

type Suggestion = { map: Record<string, string>; agreement: number; rows: number }

export function MapLabels({ name, repo, theirs, ours, onAdd }: { name: string; repo: string; theirs: string[]; ours: string[]; onAdd: (id: string) => void }) {
  const [map, setMap] = useState<Record<string, string>>({})
  const [hint, setHint] = useState<Suggestion | null>(null)
  const [asking, setAsking] = useState(false)
  const suggest = (open: boolean) => {
    if (!open || hint || asking) return
    setAsking(true)
    api<Suggestion>(`/specialists/${name}/rivals/suggest-map?repo=${encodeURIComponent(repo)}`)
      .then((s) => {
        if (!s.map) return
        setHint(s)
        setMap((m) => ({ ...s.map, ...m }))
      })
      .finally(() => setAsking(false))
  }
  const pairs = Object.entries(map).filter(([, v]) => v !== IGNORE)
  const id = `hf:${repo}#${pairs.map(([k, v]) => `${k}=${v}`).join(",")}`
  return (
    <Dialog onOpenChange={suggest}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <Shuffle />
          Map labels
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="truncate">{repo}</DialogTitle>
          <DialogDescription className="flex items-center gap-1.5">
            {asking && <Loader2 className="size-3.5 animate-spin" />}
            {asking ? "Running it on our rows to suggest a mapping" : hint ? `Suggested: agrees on ${Math.round(hint.agreement * 100)}% of ${hint.rows} of our rows` : "Pick which of our labels each of its outputs means."}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          {theirs.map((t) => (
            <div key={t} className="flex items-center gap-3">
              <span className="w-40 truncate font-mono text-sm">{t}</span>
              <Select value={map[t] ?? IGNORE} onValueChange={(v) => setMap({ ...map, [t]: v })}>
                <SelectTrigger size="sm" className="flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={IGNORE}>ignore</SelectItem>
                  {ours.map((o) => (
                    <SelectItem key={o} value={o}>
                      {o}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button disabled={!pairs.length} onClick={() => onAdd(id)}>
            <Plus />
            Add and score
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
