import { ImageUp, Loader2 } from "lucide-react"
import { useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { Draft } from "./hand-builder"
import { useScreenshot } from "./hand-screenshot"

const BLIND_LABELS = ["Small blind", "Big blind"]

export function HandTable({ draft, onChange }: { draft: Draft; onChange: (d: Draft) => void }) {
  const picker = useRef<HTMLInputElement>(null)
  const { busy, error, read } = useScreenshot(draft.blinds, onChange)
  const blind = (i: number, value: string) => onChange({ ...draft, blinds: draft.blinds.map((old, j) => (j === i ? value : old)) as [string, string] })
  const stack = (i: number, value: string) => onChange({ ...draft, stacks: draft.stacks.map((old, j) => (j === i ? value : old)) })
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <input ref={picker} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && read(e.target.files[0])} />
        <Button size="sm" variant="outline" disabled={busy} onClick={() => picker.current?.click()}>
          {busy ? <Loader2 className="animate-spin" /> : <ImageUp />}
          From a screenshot
        </Button>
        {busy && <span className="text-muted-foreground text-sm">Reading the picture, about a minute</span>}
        {error && <span className="text-destructive text-sm">{error}. Fill the hand in by hand, or try another picture.</span>}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
        {BLIND_LABELS.map((label, i) => (
          <div key={label} className="grid gap-1.5">
            <Label htmlFor={`blind-${i}`}>{label}</Label>
            <Input id={`blind-${i}`} className="tabular-nums" inputMode="decimal" value={draft.blinds[i]} onChange={(e) => blind(i, e.target.value)} />
          </div>
        ))}
        {draft.names.map((name, i) => (
          <div key={i} className="grid gap-1.5">
            <Label htmlFor={`stack-${i}`} className="truncate">
              {name} stack
            </Label>
            <Input id={`stack-${i}`} className="tabular-nums" inputMode="decimal" value={draft.stacks[i]} onChange={(e) => stack(i, e.target.value)} />
          </div>
        ))}
      </div>
    </div>
  )
}
