export const SEATS = ["SB", "BB", "UTG", "HJ", "CO", "BTN"] as const
export const STREETS = ["preflop", "flop", "turn", "river"] as const
export type Street = (typeof STREETS)[number]
export type Verb = "f" | "cc" | "cbr"
export type Step = { seat: number; verb: Verb; amount: string }
export type Draft = { hero: number; cards: string; board: Record<Exclude<Street, "preflop">, string>; steps: Record<Street, Step[]> }

export const VERBS: { verb: Verb; label: string }[] = [
  { verb: "f", label: "Fold" },
  { verb: "cc", label: "Check or call" },
  { verb: "cbr", label: "Bet or raise to" },
]

const FIRST_TO_ACT: Record<Street, number> = { preflop: 2, flop: 0, turn: 0, river: 0 }
const HIDDEN = "????"
const STACK = 10000
const BLINDS = [50, 100]

export const EMPTY: Draft = {
  hero: 5,
  cards: "",
  board: { flop: "", turn: "", river: "" },
  steps: { preflop: [], flop: [], turn: [], river: [] },
}

export const EXAMPLE: Draft = {
  hero: 1,
  cards: "Ks Qh",
  board: { flop: "4d 6s Th", turn: "", river: "" },
  steps: {
    preflop: [
      { seat: 2, verb: "f", amount: "" },
      { seat: 3, verb: "f", amount: "" },
      { seat: 4, verb: "cbr", amount: "250" },
      { seat: 5, verb: "f", amount: "" },
      { seat: 0, verb: "f", amount: "" },
      { seat: 1, verb: "cc", amount: "" },
    ],
    flop: [
      { seat: 1, verb: "cc", amount: "" },
      { seat: 4, verb: "cbr", amount: "300" },
    ],
    turn: [],
    river: [],
  },
}

const packed = (cards: string) => cards.split(" ").join("").split(",").join("")

export const dealt = (d: Draft, street: Street) => street === "preflop" || packed(d.board[street]).length > 0

function folded(d: Draft, upTo: Street): Set<number> {
  const earlier = STREETS.slice(0, STREETS.indexOf(upTo) + 1).flatMap((s) => d.steps[s])
  return new Set(earlier.filter((s) => s.verb === "f").map((s) => s.seat))
}

export function nextSeat(d: Draft, street: Street): number {
  const out = folded(d, street)
  const last = d.steps[street].at(-1)
  const from = last ? last.seat + 1 : FIRST_TO_ACT[street]
  const order = SEATS.map((_, i) => (from + i) % SEATS.length)
  return order.find((seat) => !out.has(seat)) ?? from % SEATS.length
}

const line = (s: Step) => `p${s.seat + 1} ${s.verb}${s.verb === "cbr" ? ` ${s.amount.trim()}` : ""}`

export function history(d: Draft): string {
  const deals = SEATS.map((_, i) => `d dh p${i + 1} ${i === d.hero ? packed(d.cards) : HIDDEN}`)
  const played = STREETS.filter((s) => dealt(d, s)).flatMap((s) => [...(s === "preflop" ? [] : [`d db ${packed(d.board[s])}`]), ...d.steps[s].map(line)])
  const actions = [...deals, ...played].map((a) => `'${a}'`).join(", ")
  return [
    "variant = 'NT'",
    `antes = [${SEATS.map(() => 0).join(", ")}]`,
    `blinds_or_straddles = [${[...BLINDS, ...SEATS.slice(BLINDS.length).map(() => 0)].join(", ")}]`,
    `min_bet = ${BLINDS[1]}`,
    `starting_stacks = [${SEATS.map(() => STACK).join(", ")}]`,
    `actions = [${actions}]`,
    `players = [${SEATS.map((s) => `'${s}'`).join(", ")}]`,
    "",
  ].join("\n")
}

export const ready = (d: Draft) => packed(d.cards).length === 4

export const betting = (street: Street) => `${street[0].toUpperCase()}${street.slice(1)} betting`
