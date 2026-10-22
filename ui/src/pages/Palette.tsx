import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command"
import { go } from "@/route"
import { newestFirst, titleOf } from "./specialist-groups"
import { openJob } from "@/job-sheet"

const PAGES = ["home", "specialists", "jobs", "costs", "settings"]
const JOBS: { verb: string; args: (n: string) => string[] }[] = [
  { verb: "Build", args: (n) => ["build", n] },
  { verb: "Score", args: (n) => ["eval", n] },
  { verb: "Improve", args: (n) => ["improve", n] },
  { verb: "Export", args: (n) => ["export", n] },
]

export function Palette() {
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<Summary[]>([])
  const models = items.filter((s) => s.kind !== "team").map((s) => s.name)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((o) => !o)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
  useEffect(() => {
    api<Summary[]>("/specialists")
      .then(setItems)
      .catch(() => setItems([]))
  }, [open])
  const pick = (fn: () => void) => () => {
    setOpen(false)
    fn()
  }
  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <Command>
        <CommandInput placeholder="Go to a specialist, or start a job..." />
        <CommandList>
          <CommandEmpty>Nothing matches.</CommandEmpty>
          <CommandGroup heading="Pages">
            {PAGES.map((p) => (
              <CommandItem key={p} onSelect={pick(() => go(p))}>
                {p}
              </CommandItem>
            ))}
          </CommandGroup>
          <CommandSeparator />
          <CommandGroup heading="Specialists">
            {newestFirst(items).map((s) => (
              <CommandItem key={s.name} value={`${titleOf(s)} ${s.name}`} onSelect={pick(() => go("specialists", s.name))}>
                {titleOf(s)}
                <CommandShortcut className="font-mono tracking-normal">{s.name}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
          <CommandSeparator />
          <CommandGroup heading="Jobs">
            {models.flatMap((n) =>
              JOBS.map((j) => (
                <CommandItem key={`${j.verb}-${n}`} onSelect={pick(() => runJob(j.args(n)).then((job) => openJob(job.id)))}>
                  {j.verb} {n}
                </CommandItem>
              )),
            )}
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  )
}
