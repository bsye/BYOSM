import { Copy, Search } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { useShowMore } from "@/components/show-more"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useFinished, useLive } from "@/events"
import { openJob } from "@/job-sheet"

type Idea = { title: string; prompt: string; why: string; dataset: string; rival?: string }
type Saved = { at: number | null; ideas: Idea[] }

function useIdeas(): [Saved | null, boolean] {
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "ideas")
  const running = jobs.some((j) => j.status === "running" || j.status === "queued")
  const done = useFinished()
  const [saved, setSaved] = useState<Saved | null>(null)
  useEffect(() => {
    api<Saved>("/ideas").then(setSaved, () => setSaved({ at: null, ideas: [] }))
  }, [done])
  return [saved, running]
}

export function ResearchButton() {
  const [, running] = useIdeas()
  return (
    <Button size="sm" variant="outline" disabled={running} onClick={() => runJob(["ideas"]).then((j) => openJob(j.id))}>
      <Search />
      {running ? "Researching" : "Research"}
    </Button>
  )
}

export function Ideas({ onPick }: { onPick: (prompt: string) => void }) {
  const [saved, running] = useIdeas()
  const { shown, footer } = useShowMore(saved?.ideas ?? [])
  if (!saved) return null
  if (!saved.ideas.length)
    return <p className="text-muted-foreground py-6 text-center text-sm">{running ? "Researching ideas on the web." : "No ideas yet: press Research."}</p>
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Idea</TableHead>
            <TableHead>Benchmark</TableHead>
            <TableHead className="text-right">Start</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {shown.map((i) => (
            <TableRow key={i.title}>
              <TableCell className="max-w-64">
                <div className="truncate font-medium" title={i.why}>
                  {i.title}
                </div>
                <div className="text-muted-foreground truncate text-xs">{i.rival ? `vs ${i.rival}` : i.why}</div>
              </TableCell>
              <TableCell className="text-muted-foreground max-w-40">
                <a className="block truncate hover:underline" href={`https://huggingface.co/datasets/${i.dataset}`} target="_blank" rel="noreferrer">
                  {i.dataset}
                </a>
              </TableCell>
              <TableCell className="text-right">
                <Button size="sm" variant="outline" onClick={() => onPick(i.prompt)}>
                  <Copy />
                  Use
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {footer}
    </>
  )
}
