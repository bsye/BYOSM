import { useEffect, useState } from "react"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

type Proven = { id: string; name: string; hypothesis: string; modality: string; won_on: string[]; status: "candidate" | "proven"; patch?: string }

export function ProvenMethods() {
  const [rows, setRows] = useState<Proven[]>([])
  useEffect(() => {
    api<Proven[]>("/lab/proven").then(setRows, () => setRows([]))
  }, [])
  if (!rows.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          From the lab
          <InfoTip>
            Lab methods that won a sure lead. After one win the proposer offers them to other specialists of the kind; after two a patch makes them a default
            family for you to review and apply.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Method</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Won on</TableHead>
              <TableHead>Patch</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((m) => (
              <TableRow key={m.id}>
                <TableCell className="max-w-96">
                  <div className="font-medium">{m.name}</div>
                  <div className="text-muted-foreground text-xs whitespace-normal">{m.hypothesis}</div>
                </TableCell>
                <TableCell>
                  <Badge variant={m.status === "proven" ? "default" : "outline"}>{m.status}</Badge>
                </TableCell>
                <TableCell className="text-muted-foreground text-xs">{m.won_on.join(", ")}</TableCell>
                <TableCell className="font-mono text-xs">{m.patch ? `git apply ${m.patch}` : "after a second win"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
