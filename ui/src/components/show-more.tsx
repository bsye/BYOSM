import { ChevronDown } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"

export const CARD_ROWS = 10

export function useShowMore<T>(rows: T[]): { shown: T[]; footer: React.ReactNode } {
  const [limit, setLimit] = useState(CARD_ROWS)
  const left = rows.length - limit
  const footer =
    left > 0 ? (
      <div className="-mt-4 flex justify-center">
        <Button variant="link" size="sm" className="text-muted-foreground" onClick={() => setLimit(limit + CARD_ROWS)}>
          <ChevronDown />
          {Math.min(left, CARD_ROWS)} more
        </Button>
      </div>
    ) : null
  return { shown: rows.slice(0, limit), footer }
}
