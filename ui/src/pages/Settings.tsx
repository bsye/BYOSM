import { Save } from "lucide-react"
import { useEffect, useState } from "react"
import { api, put } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"

type Role = { provider?: string; model?: string; base_url?: string }
type Config = { text: string; effective: { llm: Record<string, Role>; [k: string]: unknown } }

export function Settings() {
  const [cfg, setCfg] = useState<Config | null>(null)
  const [text, setText] = useState("")
  const [note, setNote] = useState("")
  useEffect(() => {
    api<Config>("/config").then((c) => {
      setCfg(c)
      setText(c.text)
    })
  }, [])
  const save = () =>
    put<Config>("/config", { text })
      .then((c) => {
        setCfg(c)
        setText(c.text)
        setNote("saved")
      })
      .catch((e) => setNote(String(e)))
  const roles = Object.entries(cfg?.effective.llm ?? {}).filter(([, v]) => typeof v === "object")
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>LLM roles</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>role</TableHead>
                <TableHead>provider</TableHead>
                <TableHead>model</TableHead>
                <TableHead>endpoint</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {roles.map(([name, r]) => (
                <TableRow key={name}>
                  <TableCell className="font-medium">{name}</TableCell>
                  <TableCell>{r.provider ?? "-"}</TableCell>
                  <TableCell className="font-mono text-xs">{r.model ?? "-"}</TableCell>
                  <TableCell className="font-mono text-xs">{r.base_url ?? "-"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>byosm.yaml</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea className="h-80 font-mono text-xs" value={text} onChange={(e) => setText(e.target.value)} />
          <div className="flex items-center gap-3">
            <Button onClick={save}>
              <Save />
              Save changes
            </Button>
            <span className="text-muted-foreground text-sm">{note}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
