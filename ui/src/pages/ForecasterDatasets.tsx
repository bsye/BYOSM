import { Card, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { score } from "@/format"
import type { ForecasterView } from "./forecaster-api"

export function Datasets({ f }: { f: ForecasterView }) {
  const models = Object.keys(f.overall ?? {})
  const routes = f.routes ?? {}
  const routed = Object.keys(routes).length > 0
  return (
    <Card>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Dataset</TableHead>
              {routed && <TableHead>Routed to</TableHead>}
              {models.map((m) => (
                <TableHead key={m} className="text-right">
                  {m}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(f.configs ?? {}).map(([config, row]) => (
              <TableRow key={config}>
                <TableCell className="font-medium">{config}</TableCell>
                {routed && <TableCell>{routes[config] ?? "-"}</TableCell>}
                {models.map((m) => (
                  <TableCell key={m} className="text-right tabular-nums">
                    {score(row[m]?.mase)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
