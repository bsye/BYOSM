import { ChevronDown } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { CardFooter } from "@/components/ui/card"

export const CARD_ROWS = 10

export function useShowMore<T>(rows: T[]): { shown: T[]; footer: React.ReactNode } {
  const [limit, setLimit] = useState(CARD_ROWS)
  const left = rows.length - limit
  const footer =
    left > 0 ? (
      <CardFooter className="justify-center">
        <Button variant="ghost" size="sm" onClick={() => setLimit(limit + CARD_ROWS)}>
          <ChevronDown />
          Show {Math.min(left, CARD_ROWS)} more
        </Button>
      </CardFooter>
    ) : null
  return { shown: rows.slice(0, limit), footer }
}
