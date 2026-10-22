import { Save } from "lucide-react"
import { useEffect, useState } from "react"
import { api, put } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Keys } from "./Keys"

type Role = { provider?: string; model?: string; base_url?: string }
type Config = { text: string; effective: { llm: Record<string, Role>; [k: string]: unknown } }

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
      <Tabs defaultValue="keys">
        <TabsList>
          <TabsTrigger value="keys">Keys</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>
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
