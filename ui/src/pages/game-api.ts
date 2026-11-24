import { useEffect, useState } from "react"
import { api } from "@/api"
import type { HealthReport } from "./Health"

export type Who = "Student" | "Teacher"
export type Played = { who: Who; field: string; against: string; mbb_per_hand: number; ci95: number; hands: number }
export type Street = { decisions: number; covered: number; top_move: number; mean_prob: number }
export type Agreement = { who: Who; total: Street; preflop?: Street; flop?: Street; turn?: Street; river?: Street }
export type Student = { student: string; teacher: string; model_mb: number; kl: number; top_move: number; rounds: number }
export type Game = {
  name: string
  title: string
  student: Student
  played: Played[]
  agreement: Agreement[]
  health: HealthReport | null
  blueprints: string[]
}
export type Move = { move: string; probability: number }
export type Advice = { seat: number; player: string; street: string; moves: Move[]; ms: number }

export const STREETS = ["preflop", "flop", "turn", "river"] as const

export const percent = (x: number | null | undefined) => (x == null ? "-" : `${(x * 100).toFixed(1)}%`)

export const mbb = (x: number) => `${x > 0 ? "+" : ""}${x.toFixed(1)}`

export function useGame(name: string): Game | null | undefined {
  const [game, setGame] = useState<Game | null | undefined>(undefined)
  useEffect(() => {
    api<Game>(`/games/${encodeURIComponent(name)}`)
      .then(setGame)
      .catch(() => setGame(null))
  }, [name])
  return game
}
