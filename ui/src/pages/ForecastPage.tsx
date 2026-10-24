import { Download, Loader2, Play, Trash2 } from "lucide-react"
import { del, download } from "@/api"
import { PublishDialog } from "@/components/publish-dialog"
import { go } from "@/route"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Leakage, OverTime, ScoreChart } from "./ForecastCharts"
import { ForecastToday } from "./ForecastToday"
import { ForecastTry } from "./ForecastTry"
import { runForecast, useForecasts, within, type Forecast } from "./forecast-api"

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

function Body({ f }: { f: Forecast }) {
  const r = f.result
  if (!r) return <Skeleton className="h-72 rounded-xl" />
  return (
    <>
      <div className="grid gap-4 @3xl/main:grid-cols-2">
        <Section title="AUC on test, 95% CI">
          <ScoreChart scores={r.scores} />
        </Section>
        <Section title="Each feature alone">
          <Leakage leaks={r.leakage} excluded={f.spec.excluded ?? {}} dropped={r.dropped ?? {}} />
        </Section>
      </div>
      {(r.over_time?.length ?? 0) > 1 && (
        <Section title="AUC by test year">
          <OverTime years={r.over_time ?? []} />
        </Section>
      )}
      {r.today && <ForecastTry name={f.name} shown={f.spec.shown ?? []} />}
      {r.today && (
        <Section title={`Alive on 1 Jan ${r.today.year}`}>
          <ForecastToday today={r.today} horizon={f.spec.horizon_years} check={r.scores["gradient boosting"]} testYear={f.spec.cutoffs.test[0]} />
        </Section>
      )}
    </>
  )
}

export function ForecastPage({ name }: { name: string }) {
  const [rows, running] = useForecasts()
  const f = rows?.find((r) => r.name === name)
  if (!rows) return <Skeleton className="h-96 rounded-xl" />
  if (!f)
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No forecast named {name}</EmptyTitle>
        </EmptyHeader>
      </Empty>
    )
  const test = f.result?.splits.test
  const facts = [
    within(f.spec.horizon_years),
    `tested on ${f.spec.cutoffs.test.join(", ")}`,
    test ? `${test.n.toLocaleString()} rows, ${(test.positive * 100).toFixed(1)}% ended` : "not run yet",
  ]
  const busy = running.has(name)
  const remove = async () => {
    if (window.prompt(`Type ${name} to delete this forecast`) === name) {
      await del(`/forecasts/${name}?confirm=${encodeURIComponent(name)}`)
      go("specialists")
    }
  }
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="grid gap-1">
          <h1 className="text-2xl font-semibold">{f.spec.title}</h1>
          <CardDescription>{facts.join(" · ")}</CardDescription>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" disabled={!f.result?.today} onClick={() => void download(`/forecasts/${name}/scored.csv`, `${name}-scored.csv`)}>
            <Download />
            CSV
          </Button>
          <PublishDialog
            name={name}
            exported={Boolean(f.result?.today)}
            size="sm"
            command={(repo, pub) => ["forecast", "publish", name, "--repo", repo, ...(pub ? ["--public"] : [])]}
          />
          <Button size="sm" disabled={busy} onClick={() => void runForecast(name)}>
            {busy ? <Loader2 className="animate-spin" /> : <Play />}
            {busy ? "Running" : "Run"}
          </Button>
          <Button size="icon" variant="destructive" className="size-8" aria-label="Delete this forecast" title="Delete" onClick={() => void remove()}>
            <Trash2 />
          </Button>
        </div>
      </div>
      <Body f={f} />
    </div>
  )
}
