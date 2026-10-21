import { Plus, Save, ShieldCheck, Trash2 } from "lucide-react"
import { useEffect, useState } from "react"
import { api, del, put } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { InfoTip } from "@/components/info-tip"

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
    setName("")
    load()
  }
  const rows = [...keys, ...(name && !keys.some((k) => k.name === name) ? [{ name, set: false, in_file: false }] : [])]
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          API keys
          <InfoTip>Stored in the workspace .env, readable only by you, never sent back to the browser.</InfoTip>
        </CardTitle>
        <CardAction>
          <ShieldCheck className="text-muted-foreground size-4" />
        </CardAction>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>New value</TableHead>
              <TableHead className="text-right" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((k) => (
              <TableRow key={k.name}>
                <TableCell className="font-mono text-xs">{k.name}</TableCell>
                <TableCell>
                  <Badge variant={k.set ? "secondary" : "outline"}>{k.set ? "Set" : "Not set"}</Badge>
                </TableCell>
                <TableCell>
                  <Input
                    className="h-8 max-w-sm"
                    type="password"
                    autoComplete="off"
                    placeholder="Paste a value"
                    value={draft[k.name] ?? ""}
                    onChange={(e) => setDraft({ ...draft, [k.name]: e.target.value })}
                  />
                </TableCell>
                <TableCell className="space-x-2 text-right">
                  <Button size="sm" disabled={!draft[k.name]} onClick={() => save(k.name)}>
                    <Save />
                    Save
                  </Button>
                  {k.in_file && (
                    <Button size="sm" variant="outline" onClick={() => del(`/keys/${k.name}`).then(load)}>
                      <Trash2 />
                      Remove
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="mt-3 flex items-center gap-2">
          <Plus className="text-muted-foreground size-4" />
          <Input
            className="h-8 max-w-xs font-mono text-xs"
            placeholder="ANOTHER_KEY_NAME"
            value={name}
            onChange={(e) => setName(e.target.value.toUpperCase())}
          />
        </div>
      </CardContent>
    </Card>
  )
}
