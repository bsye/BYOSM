import { useEffect, useState } from "react"
import { api, runJob, type Summary } from "@/api"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { go } from "@/route"

const PAGES = ["create", "specialists", "systems", "jobs", "keys", "settings"]
const JOBS: { verb: string; args: (n: string) => string[] }[] = [
  { verb: "Build", args: (n) => ["build", n] },
  { verb: "Score", args: (n) => ["eval", n] },
  { verb: "Improve", args: (n) => ["improve", n] },
  { verb: "Export", args: (n) => ["export", n] },
]

export function Palette() {
  const [open, setOpen] = useState(false)
  const [names, setNames] = useState<string[]>([])
  const [systems, setSystems] = useState<string[]>([])
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
    if (!open) return
    api<Summary[]>("/specialists").then((s) => setNames(s.map((x) => x.name))).catch(() => setNames([]))
    api<{ name: string }[]>("/systems").then((s) => setSystems(s.map((x) => x.name))).catch(() => setSystems([]))
  }, [open])
  const pick = (fn: () => void) => () => {
    setOpen(false)
    fn()
  }
  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
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
          {names.map((n) => (
            <CommandItem key={n} onSelect={pick(() => go("specialists", n))}>
              {n}
            </CommandItem>
          ))}
          {systems.map((n) => (
            <CommandItem key={`system-${n}`} onSelect={pick(() => go("systems"))}>
              {n} (system)
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Jobs">
          {names.flatMap((n) =>
            JOBS.map((j) => (
              <CommandItem key={`${j.verb}-${n}`} onSelect={pick(() => runJob(j.args(n)).then((job) => go("jobs", job.id)))}>
                {j.verb} {n}
              </CommandItem>
            )),
          )}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
