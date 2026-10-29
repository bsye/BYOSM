import { Download, FilePlus, ShieldCheck } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { api, download, runJob, upload, type CheckReport as Report, type DataRequest } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useFinished } from "@/events"
import { CheckReport } from "./CheckReport"
import { openJob } from "@/job-sheet"

const TABLE_ACCEPT = ".csv,.xlsx"

const open = (args: string[]) => runJob(args).then((j) => openJob(j.id))

function Needs({ counts }: { counts: NonNullable<DataRequest["request"]>["counts"] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>label</TableHead>
          <TableHead>train rows (min)</TableHead>
          <TableHead>train rows (recommended)</TableHead>
          <TableHead>test rows</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {Object.entries(counts).map(([label, c]) => (
          <TableRow key={label}>
            <TableCell className="font-medium">{label}</TableCell>
            <TableCell>{c.train_min}</TableCell>
            <TableCell>{c.train_recommended}</TableCell>
            <TableCell>{c.test_needed}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function Uploads({ name, req, onUploaded }: { name: string; req: DataRequest; onUploaded: () => void }) {
  const [note, setNote] = useState("")
  const send = async (file: File | undefined) => {
    if (!file) return
    try {
      const done = await upload<{ path: string }>(`/specialists/${name}/uploads`, file)
      setNote(`uploaded ${done.path}`)
      onUploaded()
    } catch (e) {
      setNote(String(e))
    }
  }
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Input type="file" accept={TABLE_ACCEPT} className="w-fit" onChange={(e) => send(e.target.files?.[0])} />
        <span className="text-muted-foreground text-sm">{note}</span>
      </div>
      {req.uploads.map((path) => (
        <div key={path} className="flex items-center justify-between gap-3">
          <span className="font-mono text-xs">{path}</span>
          <Button size="sm" onClick={() => open(["check-data", name, path])}>
            <ShieldCheck />
            Check
          </Button>
        </div>
      ))}
    </div>
  )
}

export function YourData({ name }: { name: string }) {
  const [req, setReq] = useState<DataRequest | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const finished = useFinished()
  const load = useCallback(() => {
    api<DataRequest>(`/specialists/${name}/request`)
      .then(setReq)
      .catch(() => setReq(null))
    api<{ report: Report | null }>(`/specialists/${name}/check`)
      .then((r) => setReport(r.report))
      .catch(() => setReport(null))
  }, [name])
  useEffect(() => {
    load()
  }, [load, finished])
  if (!req) return null
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Your data</CardTitle>
          <div className="flex flex-wrap gap-2">
            {req.request ? (
              req.templates.map((kind) => (
                <Button
                  key={kind}
                  variant="outline"
                  size="sm"
                  onClick={() => download(`/specialists/${name}/request/template.${kind}`, `${name}-template.${kind}`)}
                >
                  <Download />
                  Download template.{kind}
                </Button>
              ))
            ) : (
              <Button variant="outline" size="sm" onClick={() => open(["request-data", name])}>
                <FilePlus />
                Write a data request
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {req.request && <Needs counts={req.request.counts} />}
          <Uploads name={name} req={req} onUploaded={load} />
        </CardContent>
      </Card>
      {report && <CheckReport report={report} />}
    </>
  )
}
