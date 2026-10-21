import { useEffect, useState } from "react"
import { api, del, put } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

type Key = { name: string; set: boolean; in_file: boolean }

export function Keys() {
  const [keys, setKeys] = useState<Key[]>([])
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [name, setName] = useState("")
  const load = () => api<Key[]>("/keys").then(setKeys)
  useEffect(() => {
    load()
  }, [])
  const save = async (n: string) => {
    await put(`/keys/${n}`, { value: draft[n] ?? "" })
    setDraft({ ...draft, [n]: "" })
    load()
  }
  return (
    <div className="space-y-4">
      <Alert>
        <AlertTitle>Keys stay on this machine</AlertTitle>
        <AlertDescription>
          They are written to the workspace .env (readable only by you, never committed) and passed to jobs. The server never
          sends a key back to the browser.
        </AlertDescription>
      </Alert>
      <Card>
        <CardHeader>
          <CardTitle>API keys</CardTitle>
          <CardDescription>Used by rivals and teachers (Jev, hosted LLMs) and for the Hugging Face Hub.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {[...keys, ...(name && !keys.some((k) => k.name === name) ? [{ name, set: false, in_file: false }] : [])].map((k) => (
            <div key={k.name} className="flex flex-wrap items-center gap-3">
              <span className="w-48 font-mono text-sm">{k.name}</span>
              <Badge variant={k.set ? "default" : "outline"}>{k.set ? "set" : "not set"}</Badge>
              <Input
                className="max-w-sm"
                type="password"
                autoComplete="off"
                placeholder="paste a new value"
                value={draft[k.name] ?? ""}
                onChange={(e) => setDraft({ ...draft, [k.name]: e.target.value })}
              />
              <Button size="sm" disabled={!draft[k.name]} onClick={() => save(k.name)}>
                Save
              </Button>
              {k.in_file && (
                <Button size="sm" variant="outline" onClick={() => del(`/keys/${k.name}`).then(load)}>
                  Remove
                </Button>
              )}
            </div>
          ))}
          <Input className="max-w-sm" placeholder="ANOTHER_KEY_NAME" value={name} onChange={(e) => setName(e.target.value.toUpperCase())} />
        </CardContent>
      </Card>
    </div>
  )
}
