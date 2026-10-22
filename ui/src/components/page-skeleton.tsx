import { STAT_GRID } from "@/components/stat-card"
import { Skeleton } from "@/components/ui/skeleton"

const STAT_CARDS = 4

export function PageSkeleton() {
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <div className="space-y-2">
        <Skeleton className="h-8 w-56" />
        <Skeleton className="h-4 w-96 max-w-full" />
      </div>
      <div className={STAT_GRID}>
        {Array.from({ length: STAT_CARDS }, (_, i) => (
          <Skeleton key={i} className="h-36 rounded-xl" />
        ))}
      </div>
      <Skeleton className="h-96 rounded-xl" />
    </div>
  )
}
