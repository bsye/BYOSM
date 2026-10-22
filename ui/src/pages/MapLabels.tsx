import { Plus, Shuffle } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const IGNORE = "__ignore__"

export function MapLabels({ repo, theirs, ours, onAdd }: { repo: string; theirs: string[]; ours: string[]; onAdd: (id: string) => void }) {
  const [map, setMap] = useState<Record<string, string>>({})
  const pairs = Object.entries(map).filter(([, v]) => v !== IGNORE)
  const id = `hf:${repo}#${pairs.map(([k, v]) => `${k}=${v}`).join(",")}`
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <Shuffle />
          Map labels
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="truncate">{repo}</DialogTitle>
          <DialogDescription>Pick which of our labels each of its outputs means.</DialogDescription>
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
