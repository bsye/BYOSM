import { Combine, Search } from "lucide-react"
import { useState } from "react"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DataPreview } from "./DataPreview"
import type { KeyPair, Kept } from "./opendata-api"

function TablePick({ kept, value, onChange, label }: { kept: Kept[]; value: string; onChange: (v: string) => void; label: string }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-72" aria-label={label}>
        <SelectValue placeholder={label} />
      </SelectTrigger>
      <SelectContent>
        {kept.map((k) => (
          <SelectItem key={k.file.url} value={k.file.url}>
            {k.title} ({k.file.name})
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export function JoinBuilder({ kept }: { kept: Kept[] }) {
  const ready = kept.filter((k) => k.cached)
  const [left, setLeft] = useState("")
  const [right, setRight] = useState("")
  const [pairs, setPairs] = useState<KeyPair[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [shown, setShown] = useState<KeyPair | null>(null)
  if (ready.length < 2) return null
  const q = (extra: string) => `left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}${extra}`
  const find = () => {
    setBusy(true)
    setPairs(null)
    api<KeyPair[]>(`/opendata/join/suggest?${q("")}`)
      .then(setPairs, () => setPairs([]))
      .finally(() => setBusy(false))
  }
  const path = shown ? `/opendata/join/preview?${q(`&left_key=${encodeURIComponent(shown.left)}&right_key=${encodeURIComponent(shown.right)}`)}` : null
  const title = (url: string) => ready.find((k) => k.file.url === url)?.title ?? ""
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Join two tables
          <InfoTip>Keys are suggested by name and measured: the share of left rows whose key appears anywhere in the right table.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <TablePick kept={ready} value={left} onChange={setLeft} label="Left table" />
          <TablePick kept={ready} value={right} onChange={setRight} label="Right table" />
          <Button disabled={!left || !right || left === right || busy} onClick={find}>
            <Search />
            {busy ? "Measuring keys" : "Find keys"}
          </Button>
        </div>
        {pairs?.length === 0 && <p className="text-muted-foreground text-sm">No shared key found between these tables.</p>}
        {pairs?.map((p) => (
          <div key={`${p.left}=${p.right}`} className="flex items-center gap-3 text-sm">
            <span className="font-mono text-xs">
              {p.left} = {p.right}
            </span>
            <span className="text-muted-foreground tabular-nums">{Math.round(p.match * 100)}% of left keys match</span>
            <Button size="sm" variant="outline" className="ml-auto" onClick={() => setShown(p)}>
              <Combine />
              Preview join
            </Button>
          </div>
        ))}
      </CardContent>
      <DataPreview path={path} title={`${title(left)} joined with ${title(right)}`} onClose={() => setShown(null)} />
    </Card>
  )
}
