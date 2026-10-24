import { ArrowRight, Loader2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { href } from "@/route"
import { best, useForecasts } from "./forecast-api"
import { useAsks, useLibrary } from "./opendata-api"
import { Section } from "./Section"

const SHOWN = 5

function Open({ to, label }: { to: string; label: string }) {
  return (
    <Button size="sm" variant="ghost" asChild>
      <a href={href(to)}>
        {label}
        <ArrowRight />
      </a>
    </Button>
  )
}

function ForecastsCard() {
  const [rows, running] = useForecasts()
  return (
    <Card>
      <CardHeader>
        <CardTitle>Event risk</CardTitle>
        <CardAction>
          <Open to="specialists" label="All" />
        </CardAction>
      </CardHeader>
      <CardContent>
        {!rows ? (
          <Skeleton className="h-40 rounded-lg" />
        ) : !rows.length ? (
          <p className="text-muted-foreground text-sm">Ask a question like "will an Irish company close within 3 years?"</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Question</TableHead>
                <TableHead className="text-right">AUC</TableHead>
                <TableHead className="text-right">Expected</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.slice(0, SHOWN).map((f) => (
                <TableRow key={f.name}>
                  <TableCell className="max-w-72">
                    <a className="flex items-center gap-2 font-medium hover:underline" href={href("specialists", f.name)}>
                      <span className="truncate">{f.spec.title}</span>
                      {running.has(f.name) && <Loader2 className="text-muted-foreground size-3.5 shrink-0 animate-spin" />}
                    </a>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{best(f)?.auc.toFixed(3) ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">{f.result?.today?.expected.toLocaleString() ?? "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

function DataCard() {
  const [asks] = useAsks()
  const [kept] = useLibrary()
  return (
    <Card>
      <CardHeader>
        <CardTitle>Open data</CardTitle>
        <CardAction>
          <Open to="datasets" label="Datasets" />
        </CardAction>
      </CardHeader>
      <CardContent>
        {!asks || !kept ? (
          <Skeleton className="h-40 rounded-lg" />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Request</TableHead>
                <TableHead className="text-right">Tables</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">Kept in the library</TableCell>
                <TableCell className="text-right tabular-nums">{kept.length}</TableCell>
              </TableRow>
              {asks.slice(0, SHOWN - 1).map((a) => (
                <TableRow key={a.request}>
                  <TableCell className="max-w-72">
                    <div className="flex items-center gap-2">
                      <span className="truncate">{a.request}</span>
                      {a.status === "running" && <Loader2 className="text-muted-foreground size-3.5 shrink-0 animate-spin" />}
                      {a.status === "failed" && <Badge variant="destructive">Failed</Badge>}
                    </div>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{a.datasets.length}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

export function HomeForecasts() {
  return (
    <Section title="Event risk and data">
      <div className="grid auto-rows-fr gap-4 @5xl/main:grid-cols-2">
        <ForecastsCard />
        <DataCard />
      </div>
    </Section>
  )
}
