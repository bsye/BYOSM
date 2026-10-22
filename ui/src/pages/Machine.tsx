import { BatteryCharging, BatteryMedium, Cpu } from "lucide-react"
import { useEffect, useState } from "react"
import { api } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { gigabytes, params, rivalName, seconds } from "@/format"
import { Section } from "./Section"

const REFRESH_MS = 5000
const BILLION = 1e9
const STAT_CARDS = 4

type Backbone = { backbone: string; params: number; rows_per_second: number; runs: number; memory: number; typical_seconds: number }

export type MachineInfo = {
  cpu: { name: string; cores: number; load: number; percent: number }
  memory: { total: number; available: number; byosm: number }
  accelerator: { kind: string; name: string; memory: number }
  disk: { total: number; free: number }
  battery: { percent: number; plugged: boolean } | null
  largest: { full: number; frozen: number }
  backbones: Backbone[]
}

function useMachine(): MachineInfo | null {
  const [m, setM] = useState<MachineInfo | null>(null)
  useEffect(() => {
    const load = () => api<MachineInfo>("/machine").then(setM, () => undefined)
    load()
    const id = setInterval(load, REFRESH_MS)
    return () => clearInterval(id)
  }, [])
  return m
}

function Stat({ label, value, line, share, tip }: { label: string; value: string; line: string; share?: number; tip?: string }) {
  return (
    <Card className="@container/card h-full">
      <CardHeader>
        <CardDescription className="flex items-center gap-1">
          {label}
          {tip && <InfoTip>{tip}</InfoTip>}
        </CardDescription>
        <CardTitle className="text-2xl font-semibold tabular-nums">{value}</CardTitle>
      </CardHeader>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        {share != null && <Progress value={Math.min(100, share * 100)} />}
        <span className="text-muted-foreground line-clamp-1">{line}</span>
      </CardFooter>
    </Card>
  )
}

function BatteryBadge({ b }: { b: MachineInfo["battery"] }) {
  if (!b) return null
  return (
    <Badge variant={b.plugged || b.percent > 20 ? "outline" : "destructive"}>
      {b.plugged ? <BatteryCharging /> : <BatteryMedium />}
      {b.percent}%{b.plugged ? ", plugged in" : ""}
    </Badge>
  )
}

function Backbones({ items, budget }: { items: Backbone[]; budget: number }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Measured on this machine
          <InfoTip>Median training speed of every run so far, per encoder. Memory is an estimate for a full fine-tune with AdamW.</InfoTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Encoder</TableHead>
              <TableHead>Params</TableHead>
              <TableHead>Rows per second</TableHead>
              <TableHead>10k rows, 4 epochs</TableHead>
              <TableHead>Memory to train</TableHead>
              <TableHead>Runs</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((b) => (
              <TableRow key={b.backbone}>
                <TableCell className="font-medium">{rivalName(b.backbone)}</TableCell>
                <TableCell className="tabular-nums">{params(b.params)}</TableCell>
                <TableCell className="tabular-nums">{b.rows_per_second}</TableCell>
                <TableCell className="tabular-nums">{seconds(b.typical_seconds)}</TableCell>
                <TableCell className="tabular-nums">
                  {gigabytes(b.memory)} <span className="text-muted-foreground">({Math.max(1, Math.round((b.memory / budget) * 100))}%)</span>
                </TableCell>
                <TableCell className="text-muted-foreground tabular-nums">{b.runs}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function Badges({ m }: { m: MachineInfo }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Badge variant="outline">
        <Cpu />
        {m.cpu.name}, {m.cpu.cores} cores
      </Badge>
      <BatteryBadge b={m.battery} />
    </div>
  )
}

function Loading() {
  return (
    <>
      <div className="grid auto-rows-fr grid-cols-2 gap-4 @5xl/main:grid-cols-4">
        {Array.from({ length: STAT_CARDS }, (_, i) => (
          <Skeleton key={i} className="h-36 rounded-xl" />
        ))}
      </div>
      <Skeleton className="h-64 rounded-xl" />
    </>
  )
}

export function Machine() {
  const m = useMachine()
  return (
    <Section title="This machine" action={m && <Badges m={m} />}>
      {m ? <Stats m={m} /> : <Loading />}
    </Section>
  )
}

function Stats({ m }: { m: MachineInfo }) {
  const used = m.memory.total - m.memory.available
  const trained = Math.max(0, ...m.backbones.map((b) => b.params))
  return (
    <>
      <div className="grid auto-rows-fr grid-cols-2 gap-4 @5xl/main:grid-cols-4">
        <Stat label="CPU" value={`${Math.round(m.cpu.percent)}%`} share={m.cpu.percent / 100} line={`${m.cpu.cores} cores, load ${m.cpu.load} per core`} />
        <Stat
          label="Memory"
          value={`${gigabytes(m.memory.available)} free`}
          share={used / m.memory.total}
          line={`of ${gigabytes(m.memory.total)}, BYOSM uses ${gigabytes(m.memory.byosm)}`}
        />
        <Stat
          label="Largest fine-tune"
          value={`${(m.largest.full / BILLION).toFixed(1)}B params`}
          share={trained / m.largest.full}
          line={`largest trained so far ${params(trained)}`}
          tip={`Estimate for ${m.accelerator.name}: ${gigabytes(m.accelerator.memory)} usable for training, 16 bytes per parameter for weights, gradients and AdamW, plus 50% for activations. With a frozen encoder (4 bytes per parameter) up to ${(m.largest.frozen / BILLION).toFixed(1)}B.`}
        />
        <Stat label="Disk" value={`${gigabytes(m.disk.free)} free`} share={1 - m.disk.free / m.disk.total} line={`of ${gigabytes(m.disk.total)}`} />
      </div>
      {m.backbones.length > 0 && <Backbones items={m.backbones} budget={m.accelerator.memory} />}
    </>
  )
}
