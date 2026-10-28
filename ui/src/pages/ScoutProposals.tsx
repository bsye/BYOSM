import { Check, ExternalLink, FlaskConical, Loader2, Telescope, X } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, post, runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"
import { toast } from "sonner"

type Proposal = {
  id: string
  title: string
  effect: string
  paper: string
  kinds: string[] | null
  gain: string | null
  how: string | null
  status: "proposed" | "accepted" | "rejected"
  found: number
  published?: string
  sent?: string[]
}
type Scan = { read: number; proposed: number; kept: number }
type Scouted = { last_run: number | null; due: boolean; every_days: number; proposals: Proposal[]; last_scan?: Scan | null }

const LAB_KINDS = ["text"]
const KIND_NAMES: Record<string, string> = { text: "text classifiers", series: "forecasters", audio: "audio", image: "images", document: "documents" }
const kindNames = (kinds: string[] | null) => (kinds ?? []).map((k) => KIND_NAMES[k] ?? k).join(", ") || "this kind"
const dayOf = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })

function byDay(items: Proposal[]): [string, Proposal[]][] {
  const groups = new Map<string, Proposal[]>()
  for (const p of [...items].sort((a, b) => b.found - a.found)) groups.set(dayOf(p.found), [...(groups.get(dayOf(p.found)) ?? []), p])
  return [...groups.entries()]
}

function TryIt({ p, refresh }: { p: Proposal; refresh: () => void }) {
  const [busy, setBusy] = useState(false)
  if (!(p.kinds ?? []).some((k) => LAB_KINDS.includes(k)))
    return (
      <span className="text-muted-foreground inline-flex items-center gap-1 text-xs">
        No lab for {kindNames(p.kinds)} yet
        <InfoTip>The lab writes training code for text classifiers so far; forecasters and other kinds are next.</InfoTip>
      </span>
    )
  const run = () => {
    setBusy(true)
    post<Proposal>(`/scout/${p.id}/try`, {})
      .then((found) => toast(`Sent to the lab of ${(found.sent ?? []).join(", ")}`))
      .catch((e) => toast.error("Could not send it to a lab", { description: String(e) }))
      .finally(() => {
        setBusy(false)
        refresh()
      })
  }
  return (
    <div className="flex flex-col items-end gap-1">
      <Button size="sm" variant="outline" disabled={busy} onClick={run}>
        {busy ? <Loader2 className="animate-spin" /> : <FlaskConical />}
        Try it
      </Button>
      {p.sent?.length ? <span className="text-muted-foreground text-xs">Sent to {p.sent.join(", ")}</span> : null}
    </div>
  )
}

export function ScoutProposals() {
  const [data, setData] = useState<Scouted | null>(null)
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "scout")
  const scanning = jobs.some((j) => j.status === "running" || j.status === "queued")
  const finished = jobs.filter((j) => j.status === "done").length
  const refresh = useCallback(() => api<Scouted>("/scout").then(setData, () => setData(null)), [])
  useEffect(() => {
    refresh()
  }, [refresh, finished])
  const decide = (id: string, status: "accepted" | "rejected") => post(`/scout/${id}`, { status }).then(refresh)
  const shown = (data?.proposals ?? []).filter((p) => p.status !== "rejected")
  const scan = data?.last_scan
  const read = scan ? (scan.read ? `, ${scan.read} new papers read, ${scan.kept} of ${scan.proposed} ideas kept` : ", no new papers since the scan before") : ""
  const last = data?.last_run ? `${new Date(data.last_run * 1000).toLocaleString()}${read}` : "never"
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Proposed techniques
          <InfoTip>
            Read from recent arXiv and Hugging Face papers every {data?.every_days ?? 7} days. Accept one to send it to a specialist's lab, where an LLM
            implements it as code you approve.
          </InfoTip>
        </CardTitle>
        <CardDescription>Last scan: {last}</CardDescription>
        <CardAction>
          <Button variant="outline" disabled={scanning} onClick={() => runJob(["scout"])}>
            {scanning ? <Loader2 className="animate-spin" /> : <Telescope />}
            Find new techniques
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {shown.length === 0 ? (
          <p className="text-muted-foreground text-sm">{scanning ? "Reading recent papers..." : "No proposals yet."}</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Technique</TableHead>
                <TableHead>Kinds</TableHead>
                <TableHead>Reported gain</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {byDay(shown).flatMap(([day, items]) => [
                <TableRow key={day} className="hover:bg-transparent">
                  <TableCell colSpan={4} className="text-muted-foreground pt-4 text-xs font-medium">
                    Found {day}
                  </TableCell>
                </TableRow>,
                ...items.map((p) => (
                  <TableRow key={p.id}>
                    <TableCell className="max-w-96">
                      <div className="flex items-center gap-2 font-medium">
                        {p.title}
                        <a href={p.paper} target="_blank" rel="noreferrer" className="text-muted-foreground hover:text-foreground" aria-label="Paper">
                          <ExternalLink className="size-3" />
                        </a>
                      </div>
                      <div className="text-muted-foreground text-xs whitespace-normal">{p.effect}</div>
                      {p.published && <div className="text-muted-foreground/70 text-xs">Paper of {p.published}</div>}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {(p.kinds ?? []).map((k) => (
                          <Badge key={k} variant="outline">
                            {k}
                          </Badge>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground max-w-56 text-xs whitespace-normal">{p.gain || "-"}</TableCell>
                    <TableCell className="text-right">
                      {p.status === "accepted" ? (
                        <TryIt p={p} refresh={refresh} />
                      ) : (
                        <div className="flex justify-end gap-2">
                          <Button size="sm" variant="outline" onClick={() => decide(p.id, "rejected")}>
                            <X />
                            Reject
                          </Button>
                          <Button size="sm" onClick={() => decide(p.id, "accepted")}>
                            <Check />
                            Accept
                          </Button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                )),
              ])}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
