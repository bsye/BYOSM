import { Loader2, RefreshCw, Sparkles } from "lucide-react"
import { useState } from "react"
import { post } from "@/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

type Preview = { card: string; files: { name: string; bytes: number }[]; data: boolean }
type Edited = { card: string; refused: string | null }

const megabytes = (n: number) => `${(n / 1e6).toFixed(1)} MB`

export function PublishCard({
  name,
  repo,
  card,
  setCard,
  onPreview,
}: {
  name: string
  repo: string
  card: string
  setCard: (c: string) => void
  onPreview: (p: Preview) => void
}) {
  const [files, setFiles] = useState<Preview["files"]>([])
  const [ask, setAsk] = useState("")
  const [busy, setBusy] = useState("")
  const [note, setNote] = useState("")
  const load = () => {
    setBusy("preview")
    setNote("")
    post<Preview>(`/specialists/${name}/publish/preview`, { repo })
      .then((p) => {
        setCard(p.card)
        setFiles(p.files)
        onPreview(p)
      })
      .catch((e) => setNote(String(e)))
      .finally(() => setBusy(""))
  }
  const edit = () => {
    setBusy("edit")
    setNote("")
    post<Edited>(`/specialists/${name}/publish/edit`, { card, instruction: ask })
      .then((r) => {
        setCard(r.card)
        setNote(r.refused ? `Not applied: ${r.refused}` : "")
        if (!r.refused) setAsk("")
      })
      .catch((e) => setNote(String(e)))
      .finally(() => setBusy(""))
  }
  if (!card)
    return (
      <div className="flex flex-col items-start gap-2">
        <Button variant="outline" onClick={load} disabled={!repo.includes("/") || busy !== ""}>
          {busy === "preview" ? <Loader2 className="animate-spin" /> : <RefreshCw />}
          Preview the card
        </Button>
        {busy === "preview" && <p className="text-muted-foreground text-xs">Writing the card from the results, about a minute</p>}
        {note && <p className="text-destructive text-sm">{note}</p>}
      </div>
    )
  return (
    <div className="grid gap-3">
      <Textarea className="h-72 font-mono text-xs" value={card} onChange={(e) => setCard(e.target.value)} aria-label="Model card" />
      <div className="flex gap-2">
        <Input value={ask} placeholder="Ask the LLM to edit it, e.g. shorter overview" onChange={(e) => setAsk(e.target.value)} />
        <Button variant="outline" onClick={edit} disabled={!ask.trim() || busy !== ""}>
          {busy === "edit" ? <Loader2 className="animate-spin" /> : <Sparkles />}
          Edit
        </Button>
      </div>
      {note && <p className="text-destructive text-sm">{note}</p>}
      {files.length > 0 && (
        <p className="text-muted-foreground text-xs">
          {files.length} files, {megabytes(files.reduce((n, f) => n + f.bytes, 0))}: {files.map((f) => f.name).join(", ")}
        </p>
      )}
    </div>
  )
}
