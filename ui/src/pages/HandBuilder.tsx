import { Plus, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { betting, dealt, nextSeat, STREETS, VERBS, type Draft, type Step, type Street, type Verb } from "./hand-builder"
import { pastedImage, useScreenshot } from "./hand-screenshot"
import { HandTable } from "./HandTable"

const BOARD: { street: Exclude<Street, "preflop">; placeholder: string }[] = [
  { street: "flop", placeholder: "4d 6s Th" },
  { street: "turn", placeholder: "2c" },
  { street: "river", placeholder: "9h" },
]

function Action({ step, names, onChange, onRemove }: { step: Step; names: string[]; onChange: (s: Step) => void; onRemove: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Select value={String(step.seat)} onValueChange={(v) => onChange({ ...step, seat: Number(v) })}>
        <SelectTrigger size="sm" className="w-24" aria-label="Seat">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {names.map((seat, i) => (
            <SelectItem key={i} value={String(i)}>
              {seat}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={step.verb} onValueChange={(v) => onChange({ ...step, verb: v as Verb })}>
        <SelectTrigger size="sm" className="w-40" aria-label="Move">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {VERBS.map((v) => (
            <SelectItem key={v.verb} value={v.verb}>
              {v.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {step.verb === "cbr" && (
        <Input
          className="h-8 w-24 tabular-nums"
          inputMode="numeric"
          placeholder="chips"
          aria-label="Chips"
          value={step.amount}
          onChange={(e) => onChange({ ...step, amount: e.target.value })}
        />
      )}
      <Button size="icon" variant="ghost" className="size-8" aria-label="Remove this action" onClick={onRemove}>
        <X />
      </Button>
    </div>
  )
}

function Betting({ draft, street, onChange }: { draft: Draft; street: Street; onChange: (d: Draft) => void }) {
  const steps = draft.steps[street]
  const set = (next: Step[]) => onChange({ ...draft, steps: { ...draft.steps, [street]: next } })
  return (
    <div className="grid gap-2">
      <Label>{betting(street)}</Label>
      {steps.map((step, i) => (
        <Action
          key={i}
          step={step}
          names={draft.names}
          onChange={(s) => set(steps.map((old, j) => (j === i ? s : old)))}
          onRemove={() => set(steps.filter((_, j) => j !== i))}
        />
      ))}
      <div>
        <Button size="sm" variant="outline" onClick={() => set([...steps, { seat: nextSeat(draft, street), verb: "f", amount: "" }])}>
          <Plus />
          Action
        </Button>
      </div>
    </div>
  )
}

export function HandBuilder({ draft, onChange }: { draft: Draft; onChange: (d: Draft) => void }) {
  const { read } = useScreenshot(draft.blinds, onChange)
  const paste = (e: React.ClipboardEvent) => {
    const image = pastedImage(e.clipboardData.items)
    if (image) read(image)
  }
  return (
    <div className="grid gap-4" onPaste={paste}>
      <HandTable draft={draft} onChange={onChange} />
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <Label>Your seat</Label>
          <ToggleGroup type="single" size="sm" variant="outline" value={String(draft.hero)} onValueChange={(v) => v && onChange({ ...draft, hero: Number(v) })}>
            {draft.names.map((seat, i) => (
              <ToggleGroupItem key={i} value={String(i)}>
                {seat}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor="hand-cards">Your cards</Label>
          <Input id="hand-cards" placeholder="As Kd" value={draft.cards} onChange={(e) => onChange({ ...draft, cards: e.target.value })} />
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        {BOARD.map(({ street, placeholder }) => (
          <div key={street} className="grid gap-1.5">
            <Label htmlFor={`hand-${street}`} className="capitalize">
              {street}
            </Label>
            <Input
              id={`hand-${street}`}
              placeholder={placeholder}
              value={draft.board[street]}
              onChange={(e) => onChange({ ...draft, board: { ...draft.board, [street]: e.target.value } })}
            />
          </div>
        ))}
      </div>
      {STREETS.filter((s) => dealt(draft, s)).map((street) => (
        <Betting key={street} draft={draft} street={street} onChange={onChange} />
      ))}
    </div>
  )
}
