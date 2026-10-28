import { FileUp, Loader2 } from "lucide-react"
import { InfoTip } from "@/components/info-tip"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { usePageReading, type PageRead } from "./page-reading"

const TOP = 3
const SHADE_MAX = 80
const MIN_SCALE = 0.01
const ACCEPT = ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp"
const pct = (v: number) => `${Math.round(v * 100)}%`

function heat(drop: number, widest: number): string {
  const strength = Math.round((Math.max(0, drop) / widest) * SHADE_MAX)
  return `color-mix(in oklch, var(--chart-2) ${strength}%, transparent)`
}

function Heatmap({ page, rows }: { page: PageRead; rows: Record<number, number[]> }) {
  const drops = Object.values(rows).flat()
  const widest = Math.max(MIN_SCALE, ...drops)
  const done = Object.keys(rows).length === page.grid
  return (
    <div className="grid gap-2">
      <div className="relative aspect-square w-full max-w-sm overflow-hidden rounded-md border">
        <img src={page.page} alt="First page" className="size-full object-cover" />
        <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${page.grid}, 1fr)` }}>
          {Array.from({ length: page.grid * page.grid }, (_, i) => {
            const drop = rows[Math.floor(i / page.grid)]?.[i % page.grid]
            return drop == null ? (
              <div key={i} className="bg-muted/40 animate-pulse" />
            ) : (
              <div key={i} title={`${page.label} ${pct(drop)} lower without this area`} style={{ backgroundColor: heat(drop, widest) }} />
            )
          })}
        </div>
      </div>
      {done && (
        <p className="text-muted-foreground text-xs tabular-nums">
          Covering one square lowers {page.label} by at most {(Math.max(0, ...drops) * 100).toFixed(1)} points
        </p>
      )}
    </div>
  )
}

function Answer({ page }: { page: PageRead }) {
  const top = Object.entries(page.probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP)
  return (
    <div className="grid content-start gap-3">
      <span className="text-3xl font-semibold">{page.label}</span>
      {top.map(([label, p]) => (
        <div key={label} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className={label === page.label ? "font-medium" : "text-muted-foreground"}>{label}</span>
            <span className="text-foreground/70 tabular-nums">{p.toFixed(3)}</span>
          </div>
          <Progress value={p * 100} />
        </div>
      ))}
      {page.text && <p className="text-muted-foreground line-clamp-6 text-xs">{page.text}</p>}
    </div>
  )
}

export function PageTry({ name }: { name: string }) {
  const { page, rows, busy, error, send } = usePageReading(name)
  const picker = <Input type="file" accept={ACCEPT} className="w-64" onChange={(e) => e.target.files?.[0] && send(e.target.files[0])} />
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          How it reads a page
          <InfoTip>
            Each square is covered in grey and the page is read again with the same text: the darker the square, the more the answer falls without that area of
            the picture.
          </InfoTip>
        </CardTitle>
        <CardAction className="flex items-center gap-2">
          {busy && <Loader2 className="text-muted-foreground size-4 animate-spin" />}
          {page && picker}
        </CardAction>
      </CardHeader>
      <CardContent>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {page ? (
          <div className="grid gap-6 @3xl/main:grid-cols-2">
            <Heatmap page={page} rows={rows} />
            <Answer page={page} />
          </div>
        ) : (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <FileUp />
              </EmptyMedia>
              <EmptyTitle>Upload a document</EmptyTitle>
              <EmptyDescription>A PDF or a scan</EmptyDescription>
            </EmptyHeader>
            {picker}
          </Empty>
        )}
      </CardContent>
    </Card>
  )
}
