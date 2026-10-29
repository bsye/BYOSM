import { InfoTip } from "@/components/info-tip"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { signed, signedInterval } from "@/format"
import { ForestChart, type ForestRow } from "./ForestChart"

type Summary = { measured: number; mean: number | null; lo: number | null; hi: number | null }
export type Effect = Summary & { left_out: number; kinds: (Summary & { modality: string; task_type: string })[] }
type Family = { title: string; calib_effect: Effect }
type Row = ForestRow & { effect: Effect }

const rowOf = (f: Family): Row | null => {
  const { mean, lo, hi } = f.calib_effect
  return mean == null || lo == null || hi == null ? null : { title: f.title, mean, lo, hi, effect: f.calib_effect }
}

function ReadoutRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex w-full justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground font-mono font-medium tabular-nums">{value}</span>
    </div>
  )
}

function Readout({ effect }: { effect: Effect }) {
  return (
    <div className="grid w-full gap-1">
      <ReadoutRow label="Mean" value={signed(effect.mean)} />
      <ReadoutRow label="95% interval" value={signedInterval(effect.lo, effect.hi)} />
      <ReadoutRow label="Experiments" value={String(effect.measured)} />
      {effect.left_out > 0 && <ReadoutRow label="Left out" value={String(effect.left_out)} />}
      {effect.kinds.length > 1 &&
        effect.kinds.map((k) => (
          <ReadoutRow key={`${k.modality}-${k.task_type}`} label={`${k.modality} ${k.task_type}`} value={`${signed(k.mean)} (${k.measured})`} />
        ))}
    </div>
  )
}

export function EffectChart({ families: all }: { families: Family[] }) {
  const families = all.filter((f) => f.calib_effect)
  const rows = families
    .map(rowOf)
    .filter((r): r is Row => r != null)
    .sort((a, b) => b.mean - a.mean)
  const few = families.filter((f) => f.calib_effect.lo == null && f.calib_effect.measured + f.calib_effect.left_out > 0)
  if (!rows.length && !few.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Calib effect per family
          <InfoTip>
            Mean change in calib score against the experiment each one was built from, scored on the same suite version, with a 95% Student t interval across
            experiments. Experiments without such a parent are left out.
          </InfoTip>
        </CardTitle>
      </CardHeader>
      {rows.length > 0 && (
        <CardContent>
          <ForestChart rows={rows} label="Calib effect" readout={(r) => <Readout effect={r.effect} />} />
        </CardContent>
      )}
      {few.length > 0 && (
        <CardFooter className="text-muted-foreground text-xs">
          Not enough runs: {few.map((f) => `${f.title} (${f.calib_effect.measured} measured)`).join(", ")}
        </CardFooter>
      )}
    </Card>
  )
}
