import { Label } from "@/components/ui/label"
import { Pick } from "./NewForecast"
import type { Column, Kept } from "./opendata-api"

export type Events = { url: string; key: string; date: string; own: string }
export const NO_EVENTS: Events = { url: "", key: "", date: "", own: "" }
const NONE = "none"
const DATE_HINTS = ["date", "_dt", "received", "filed"]

const isDate = (c: Column) => DATE_HINTS.some((h) => c.name.toLowerCase().includes(h))

export function guessed(e: Events, columns: Column[], main: Column[]): Events {
  const shared = columns.find((c) => main.some((m) => m.name === c.name))?.name ?? ""
  return { url: e.url, key: e.key || shared, own: e.own || shared, date: e.date || (columns.find(isDate)?.name ?? "") }
}

export function eventsSpec(e: Events, title: string) {
  if (!e.url || !e.key || !e.date || !e.own) return {}
  const name = title.toLowerCase().split(" ").filter(Boolean).join("_") || "events"
  return { key: e.own, histories: { [name]: { sources: [e.url], key: e.key, date: e.date } } }
}

type Props = { tables: Kept[]; main: Column[]; columns: Column[]; value: Events; onChange: (e: Events) => void }

export function EventsPick({ tables, main, columns, value, onChange }: Props) {
  const options = (cs: Column[]) => cs.map((c) => ({ value: c.name, label: c.name }))
  return (
    <div className="grid gap-3 rounded-md border p-3">
      <Label className="text-muted-foreground text-xs">Events before each cutoff (optional)</Label>
      <Pick label="Event table" value={value.url || NONE} onChange={(v) => onChange({ ...NO_EVENTS, url: v === NONE ? "" : v })} options={[{ value: NONE, label: "None" }, ...tables.map((k) => ({ value: k.file.url, label: k.title }))]} />
      {value.url && (
        <div className="grid gap-3 sm:grid-cols-3">
          <Pick label="Matches" value={value.own} onChange={(own) => onChange({ ...value, own })} options={options(main)} />
          <Pick label="Event key" value={value.key} onChange={(key) => onChange({ ...value, key })} options={options(columns)} />
          <Pick label="Event date" value={value.date} onChange={(date) => onChange({ ...value, date })} options={options(columns)} />
        </div>
      )}
    </div>
  )
}
