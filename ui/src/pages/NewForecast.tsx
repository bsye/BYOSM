import { Plus } from "lucide-react"
import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api, post, runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { openJob } from "@/job-sheet"
import { filePreview } from "./DataPreview"
import { EventsPick, eventsSpec, guessed, NO_EVENTS, type Events } from "./EventsPick"
import { useLibrary, type Column, type Preview } from "./opendata-api"

const HORIZONS = [1, 2, 3, 5, 10]
const BACKTEST_YEARS = 5
const BACKTEST_STEP = 2
const YEARS_OFFERED = 10
const ID_SHARE = 0.9
const DATE_HINTS = ["date", "_dt", "since", "until", "start", "end"]

const slug = (t: string) => t.toLowerCase().split("").map((c) => (c >= "a" && c <= "z") || (c >= "0" && c <= "9") ? c : "-").join("").split("-").filter(Boolean).join("-")
const isDate = (c: Column) => DATE_HINTS.some((h) => c.name.toLowerCase().includes(h))
const isId = (c: Column, sampled: number) => c.distinct >= sampled * ID_SHARE


export function Pick({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <div className="grid gap-1.5">
      <Label>{label}</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Choose" />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export function useColumns(url: string): Preview | null {
  const [preview, setPreview] = useState<Preview | null>(null)
  useEffect(() => {
    setPreview(null)
    if (url) api<Preview>(filePreview(url)).then(setPreview, () => setPreview(null))
  }, [url])
  return preview
}

export function NewForecast({ onSaved }: { onSaved: () => void }) {
  const [kept] = useLibrary()
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState("")
  const [title, setTitle] = useState("")
  const [start, setStart] = useState("")
  const [end, setEnd] = useState("")
  const [horizon, setHorizon] = useState("5")
  const thisYear = new Date().getFullYear()
  const [test, setTest] = useState(String(thisYear - 5))
  const [features, setFeatures] = useState<string[]>([])
  const [picked, setEvents] = useState<Events>(NO_EVENTS)
  const preview = useColumns(url)
  const eventColumns = useColumns(picked.url)?.columns ?? []
  const columns = preview?.columns ?? []
  const events = guessed(picked, eventColumns, columns)
  useEffect(() => {
    if (!preview) return
    const dates = [...preview.columns.filter(isDate)].sort((a, b) => a.missing - b.missing)
    setStart(dates[0]?.name ?? "")
    setEnd(dates.filter((d) => d.missing < 1).at(-1)?.name ?? "")
    setFeatures(preview.columns.filter((c) => !isDate(c) && !isId(c, preview.sampled)).map((c) => c.name))
  }, [preview])
  const latest = thisYear - Number(horizon)
  const years = Array.from({ length: YEARS_OFFERED }, (_, i) => String(latest - i))
  const chosen = features.filter((f) => f !== start && f !== end)
  const ready = url && title && start && end && start !== end && chosen.length && Number(test) <= latest
  const save = () => {
    const name = slug(title)
    const spec = { title, source: url, start, event: end, horizon_years: Number(horizon), test_year: Number(test), backtest: Array.from({ length: BACKTEST_YEARS }, (_, i) => latest - i * BACKTEST_STEP), features: chosen, ...eventsSpec(events, tables.find((k) => k.file.url === events.url)?.title ?? "") }
    post("/forecasts", { name, spec })
      .then(() => runJob(["forecast", "run", name]))
      .then((j) => {
        setOpen(false)
        onSaved()
        openJob(j.id)
      }, (e) => toast.error(String(e)))
  }
  const tables = (kept ?? []).filter((k) => k.cached)
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" disabled={!tables.length}>
          <Plus />
          New forecast
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>New forecast</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid gap-1.5">
            <Label htmlFor="forecast-title">Question</Label>
            <Input id="forecast-title" value={title} placeholder="Will this company close within 5 years?" onChange={(e) => setTitle(e.target.value)} />
          </div>
          <Pick label="Table" value={url} onChange={setUrl} options={tables.map((k) => ({ value: k.file.url, label: k.title }))} />
          <div className="grid gap-4 sm:grid-cols-2">
            <Pick label="Starts on" value={start} onChange={setStart} options={columns.map((c) => ({ value: c.name, label: c.name }))} />
            <Pick label="Ends on" value={end} onChange={setEnd} options={columns.map((c) => ({ value: c.name, label: c.name }))} />
            <Pick label="Within" value={horizon} onChange={setHorizon} options={HORIZONS.map((h) => ({ value: String(h), label: h === 1 ? "1 year" : `${h} years` }))} />
            <Pick label="Test on" value={test} onChange={setTest} options={years.map((y) => ({ value: y, label: `alive on 1 Jan ${y}` }))} />
          </div>
          {columns.length > 0 && <EventsPick tables={tables.filter((k) => k.file.url !== url)} main={columns} columns={eventColumns} value={events} onChange={setEvents} />}
          {columns.length > 0 && (
            <div className="grid gap-1.5">
              <Label>Features</Label>
              <ScrollArea className="h-40 rounded-md border p-3">
                <div className="grid gap-2 sm:grid-cols-2">
                  {columns
                    .filter((c) => c.name !== start && c.name !== end)
                    .map((c) => (
                      <label key={c.name} className="flex items-center gap-2 text-sm">
                        <Checkbox checked={features.includes(c.name)} onCheckedChange={(on) => setFeatures(on ? [...features, c.name] : features.filter((f) => f !== c.name))} />
                        <span className="truncate font-mono text-xs">{c.name}</span>
                      </label>
                    ))}
                </div>
              </ScrollArea>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button disabled={!ready} onClick={save}>
            Save and run
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
