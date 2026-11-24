import { STAT_GRID, StatCard } from "@/components/stat-card"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { GameTry } from "./GameTry"
import { Health } from "./Health"
import { mbb, percent, STREETS, type Agreement, type Game, type Played } from "./game-api"

const interval = (r: Played) => `95% CI ${mbb(r.mbb_per_hand - r.ci95)} to ${mbb(r.mbb_per_hand + r.ci95)}`

function Stats({ g }: { g: Game }) {
  const { student } = g
  const seated = g.played.find((r) => r.who === "Student" && r.field === student.teacher)
  const ours = g.agreement.find((a) => a.who === "Student")
  const theirs = g.agreement.find((a) => a.who === "Teacher")
  return (
    <div className={STAT_GRID}>
      <StatCard
        label="Against five teacher copies"
        value={seated ? mbb(seated.mbb_per_hand) : "-"}
        detail={seated ? `mbb/hand, ${interval(seated)}, ${seated.hands.toLocaleString()} hands` : "not seated yet"}
      />
      <StatCard
        label="Pluribus's move matched"
        value={percent(ours?.total.top_move)}
        detail={theirs ? `its teacher ${percent(theirs.total.top_move)}` : "its teacher is not scored yet"}
      />
      <StatCard label="Teacher's top move kept" value={percent(student.top_move)} detail={`held-out KL ${student.kl.toFixed(4)}`} />
      <StatCard label="Size" value={`${student.model_mb.toFixed(1)} MB`} detail="a student of its self-play teacher" />
    </div>
  )
}

function AtTheTable({ rows }: { rows: Played[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>At a six-player table</CardTitle>
        <CardDescription>Thousandths of a big blind won per hand</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Player</TableHead>
              <TableHead>Other five seats</TableHead>
              <TableHead className="text-right">mbb/hand</TableHead>
              <TableHead className="text-right">95% CI</TableHead>
              <TableHead className="text-right">Hands</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={`${r.who} ${r.against}`}>
                <TableCell className="font-medium">{r.who}</TableCell>
                <TableCell>{r.against}</TableCell>
                <TableCell className="text-right font-medium tabular-nums">{mbb(r.mbb_per_hand)}</TableCell>
                <TableCell className="text-muted-foreground text-right tabular-nums">
                  {mbb(r.mbb_per_hand - r.ci95)} to {mbb(r.mbb_per_hand + r.ci95)}
                </TableCell>
                <TableCell className="text-right tabular-nums">{r.hands.toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function WithPluribus({ rows }: { rows: Agreement[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pluribus's logged moves</CardTitle>
        <CardDescription>Share of {rows[0].total.decisions.toLocaleString()} decisions where the top move is the one Pluribus made</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Player</TableHead>
              <TableHead className="text-right">All</TableHead>
              {STREETS.map((s) => (
                <TableHead key={s} className="text-right capitalize">
                  {s}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((a) => (
              <TableRow key={a.who}>
                <TableCell className="font-medium">{a.who}</TableCell>
                <TableCell className="text-right font-medium tabular-nums">{percent(a.total.top_move)}</TableCell>
                {STREETS.map((s) => (
                  <TableCell key={s} className="text-right tabular-nums">
                    {percent(a[s]?.top_move)}
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

export function GamePage({ g }: { g: Game }) {
  const rounds = g.student.rounds ? `, ${g.student.rounds} rounds on its own play` : ""
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="grid gap-1">
        <h1 className="text-2xl font-semibold">{g.title}</h1>
        <CardDescription>
          <span className="font-mono text-xs">{g.name}</span> · six seats, 100 big blinds{rounds}
        </CardDescription>
      </div>
      <Stats g={g} />
      <GameTry name={g.name} />
      <Health h={g.health} />
      {g.played.length > 0 && <AtTheTable rows={g.played} />}
      {g.agreement.length > 0 && <WithPluribus rows={g.agreement} />}
    </div>
  )
}
