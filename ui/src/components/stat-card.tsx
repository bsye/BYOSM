import type { ReactNode } from "react"
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export function StatCard({ label, value, detail }: { label: string; value: string; detail?: ReactNode }) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl font-semibold tabular-nums">{value}</CardTitle>
      </CardHeader>
      {detail && <CardFooter className="text-muted-foreground mt-auto hidden text-xs @xl/main:flex">{detail}</CardFooter>}
    </Card>
  )
}

export const STAT_GRID = "grid auto-rows-fr grid-cols-2 gap-4 @5xl/main:grid-cols-4"
