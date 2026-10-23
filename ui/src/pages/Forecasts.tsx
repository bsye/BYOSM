import { Loader2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { href } from "@/route"
import { best, useForecasts, type Forecast } from "./forecast-api"
import { NewForecast } from "./NewForecast"

const SKELETONS = 3

export const within = (years: number) => (years === 1 ? "within a year" : `within ${years} years`)

function ForecastTile({ f, running }: { f: Forecast; running: boolean }) {
  const score = best(f)
  return (
    <a href={href("forecasts", f.name)} className="block">
      <Card className="hover:bg-muted/40 h-full transition-colors">
        <CardHeader>
          <CardDescription>{within(f.spec.horizon_years)}</CardDescription>
          <CardTitle className="line-clamp-2">{f.spec.title}</CardTitle>
          <CardAction>{running ? <Loader2 className="text-muted-foreground size-4 animate-spin" /> : !f.result && <Badge variant="outline">Not run</Badge>}</CardAction>
        </CardHeader>
        <CardFooter className="mt-auto grid grid-cols-2 gap-2 text-sm">
          <div>
            <div className="text-muted-foreground text-xs">AUC</div>
            <div className="text-lg font-semibold tabular-nums">{score ? score.auc.toFixed(3) : "—"}</div>
          </div>
          <div>
            <div className="text-muted-foreground text-xs">Expected to end</div>
            <div className="text-lg font-semibold tabular-nums">{f.result?.today ? f.result.today.expected.toLocaleString() : "—"}</div>
          </div>
        </CardFooter>
      </Card>
    </a>
  )
}

export function Forecasts() {
  const [rows, running, reload] = useForecasts()
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="flex items-center justify-between gap-2">
        <h1 className="text-2xl font-semibold">Forecasts</h1>
        <NewForecast onSaved={reload} />
      </div>
      {rows?.length === 0 && (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>No forecasts yet</EmptyTitle>
            <EmptyDescription>Keep a table with start and end dates in Datasets, then create one.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      )}
      <div className="grid auto-rows-fr gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
        {rows ? rows.map((f) => <ForecastTile key={f.name} f={f} running={running.has(f.name)} />) : Array.from({ length: SKELETONS }, (_, i) => <Skeleton key={i} className="h-36 rounded-xl" />)}
      </div>
    </div>
  )
}
