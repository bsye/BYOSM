import { Save } from "lucide-react"
import { useEffect, useState } from "react"
import { api, put } from "@/api"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { modelName } from "@/jobs"
import { Keys } from "./Keys"

type Role = { provider?: string; model?: string; base_url?: string }
type Config = { text: string; effective: { llm: Record<string, Role>; [k: string]: unknown } }

function Llm({ cfg, onSaved }: { cfg: Config | null; onSaved: (c: Config) => void }) {
  const llm = useLlmChoice()
  const [note, setNote] = useState("")
  const save = () => {
    const config = llm.config()
    if (!config) return
    put<Config>("/config/default-llm", { config })
      .then((c) => {
        onSaved(c)
        setNote("Saved")
      })
      .catch((e) => setNote(String(e)))
  }
  const roles = Object.entries(cfg?.effective.llm ?? {}).filter(([, v]) => typeof v === "object")
  return (
    <Card>
      <CardHeader>
        <CardTitle>Default LLM</CardTitle>
        <CardAction className="text-muted-foreground text-sm">{note}</CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <LlmPicker llm={llm} label="Default LLM" />
          <Button size="sm" disabled={!llm.choice} onClick={save}>
            <Save />
            Use for every role
          </Button>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Role</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Provider</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {roles.map(([name, r]) => (
              <TableRow key={name}>
                <TableCell className="font-medium">{name}</TableCell>
                <TableCell>{modelName(r.model ?? "-")}</TableCell>
                <TableCell className="text-foreground/70">{r.provider ?? "-"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Advanced({ cfg, onSaved }: { cfg: Config | null; onSaved: (c: Config) => void }) {
  const [text, setText] = useState(cfg?.text ?? "")
  const [note, setNote] = useState("")
  useEffect(() => setText(cfg?.text ?? ""), [cfg?.text])
  const save = () =>
    put<Config>("/config", { text })
      .then((c) => {
        onSaved(c)
        setNote("Saved")
      })
      .catch((e) => setNote(String(e)))
  return (
    <Card>
      <CardHeader>
        <CardTitle>byosm.yaml</CardTitle>
        <CardAction className="text-muted-foreground text-sm">{note}</CardAction>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea className="h-80 font-mono text-xs" value={text} onChange={(e) => setText(e.target.value)} />
        <Button onClick={save}>
          <Save />
          Save changes
        </Button>
      </CardContent>
    </Card>
  )
}

export function Settings() {
  const [cfg, setCfg] = useState<Config | null>(null)
  useEffect(() => {
    api<Config>("/config").then(setCfg)
  }, [])
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <Tabs defaultValue="llm">
        <TabsList>
          <TabsTrigger value="llm">LLM</TabsTrigger>
          <TabsTrigger value="keys">Keys</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>
        <TabsContent value="llm">
          <Llm cfg={cfg} onSaved={setCfg} />
        </TabsContent>
        <TabsContent value="keys">
          <Keys />
        </TabsContent>
        <TabsContent value="advanced">
          <Advanced cfg={cfg} onSaved={setCfg} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
