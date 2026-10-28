import { Check, ExternalLink, FlaskConical, Loader2, Telescope, X } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, post, runJob, type Summary } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useLive } from "@/events"

type Proposal = {
  id: string
  title: string
  effect: string
  paper: string
  kinds: string[] | null
  gain: string | null
  how: string | null
  status: "proposed" | "accepted" | "rejected"
}
type Scouted = { last_run: number | null; due: boolean; every_days: number; proposals: Proposal[] }

const labCapable = (s: Summary) => s.modality === "text" && s.task_type === "classify" && !s.archived

function SendToLab({ p, specialists }: { p: Proposal; specialists: Summary[] }) {
  const [picked, setPicked] = useState("")
  const ask = `Implement "${p.title}" from ${p.paper}: ${p.how ?? p.effect}`
  return (
    <div className="flex items-center gap-2">
      <Select value={picked} onValueChange={setPicked}>
        <SelectTrigger size="sm" className="w-48" aria-label="Specialist">
          <SelectValue placeholder="Specialist" />
        </SelectTrigger>
        <SelectContent>
          {specialists.map((s) => (
            <SelectItem key={s.name} value={s.name}>
              {s.title ?? s.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button size="sm" variant="outline" disabled={!picked} onClick={() => runJob(["lab", "write", picked, "--ask", ask])}>
        <FlaskConical />
        To the lab
      </Button>
    </div>
  )
}

export function ScoutProposals() {
  const [data, setData] = useState<Scouted | null>(null)
  const [specialists, setSpecialists] = useState<Summary[]>([])
  const jobs = Object.values(useLive().jobs).filter((j) => j.args[0] === "scout")
  const scanning = jobs.some((j) => j.status === "running" || j.status === "queued")
  const finished = jobs.filter((j) => j.status === "done").length
  const refresh = useCallback(() => api<Scouted>("/scout").then(setData, () => setData(null)), [])
  useEffect(() => {
    refresh()
  }, [refresh, finished])
  useEffect(() => {
    api<Summary[]>("/specialists").then(
      (all) => setSpecialists(all.filter(labCapable)),
      () => setSpecialists([]),
    )
  }, [])
  const decide = (id: string, status: "accepted" | "rejected") => post(`/scout/${id}`, { status }).then(refresh)
  const shown = (data?.proposals ?? []).filter((p) => p.status !== "rejected").sort((a, b) => Number(a.status === "accepted") - Number(b.status === "accepted"))
  const last = data?.last_run ? new Date(data.last_run * 1000).toLocaleDateString() : "never"
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
              {shown.map((p) => (
                <TableRow key={p.id}>
                  <TableCell className="max-w-96">
                    <div className="flex items-center gap-2 font-medium">
                      {p.title}
                      <a href={p.paper} target="_blank" rel="noreferrer" className="text-muted-foreground hover:text-foreground" aria-label="Paper">
                        <ExternalLink className="size-3" />
                      </a>
                    </div>
                    <div className="text-muted-foreground text-xs whitespace-normal">{p.effect}</div>
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
                      <SendToLab p={p} specialists={specialists} />
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
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
