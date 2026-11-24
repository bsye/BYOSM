export const SEATS = ["SB", "BB", "UTG", "HJ", "CO", "BTN"] as const
export const STREETS = ["preflop", "flop", "turn", "river"] as const
export type Street = (typeof STREETS)[number]
export type Verb = "f" | "cc" | "cbr"
export type Step = { seat: number; verb: Verb; amount: string }
export type Draft = {
  hero: number
  cards: string
  board: Record<Exclude<Street, "preflop">, string>
  steps: Record<Street, Step[]>
  names: string[]
  stacks: string[]
  blinds: [string, string]
}
export type Seen = { hero: number; cards: string[]; board: string[]; names: string[]; stacks: number[]; in_hand: boolean[]; said: string[] }

export const VERBS: { verb: Verb; label: string }[] = [
  { verb: "f", label: "Fold" },
  { verb: "cc", label: "Check or call" },
  { verb: "cbr", label: "Bet or raise to" },
]

const FIRST_AFTER_BLINDS = 2
const HIDDEN = "????"
const STACK = "10000"
const BLINDS: [string, string] = ["50", "100"]
const SAID: Record<string, Verb> = { fold: "f", call: "cc", check: "cc", raise: "cbr", bet: "cbr" }
const NO_STEPS = { preflop: [], flop: [], turn: [], river: [] }

export const EMPTY: Draft = {
  hero: 5,
  cards: "",
  board: { flop: "", turn: "", river: "" },
  steps: NO_STEPS,
  names: [...SEATS],
  stacks: SEATS.map(() => STACK),
  blinds: BLINDS,
}

export const EXAMPLE: Draft = {
  ...EMPTY,
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

const firstToAct = (d: Draft, street: Street) => (street === "preflop" ? FIRST_AFTER_BLINDS % d.names.length : 0)

function folded(d: Draft, upTo: Street): Set<number> {
  const earlier = STREETS.slice(0, STREETS.indexOf(upTo) + 1).flatMap((s) => d.steps[s])
  return new Set(earlier.filter((s) => s.verb === "f").map((s) => s.seat))
}

export function nextSeat(d: Draft, street: Street): number {
  const out = folded(d, street)
  const last = d.steps[street].at(-1)
  const from = last ? last.seat + 1 : firstToAct(d, street)
  const order = d.names.map((_, i) => (from + i) % d.names.length)
  return order.find((seat) => !out.has(seat)) ?? from % d.names.length
}

const line = (s: Step) => `p${s.seat + 1} ${s.verb}${s.verb === "cbr" ? ` ${s.amount.trim()}` : ""}`

export function history(d: Draft): string {
  const seats = d.names
  const deals = seats.map((_, i) => `d dh p${i + 1} ${i === d.hero ? packed(d.cards) : HIDDEN}`)
  const played = STREETS.filter((s) => dealt(d, s)).flatMap((s) => [...(s === "preflop" ? [] : [`d db ${packed(d.board[s])}`]), ...d.steps[s].map(line)])
  const actions = [...deals, ...played].map((a) => `'${a}'`).join(", ")
  const blinds = [...d.blinds, ...seats.slice(d.blinds.length).map(() => "0")]
  return [
    "variant = 'NT'",
    `antes = [${seats.map(() => 0).join(", ")}]`,
    `blinds_or_straddles = [${blinds.map((b) => b.trim()).join(", ")}]`,
    `min_bet = ${d.blinds[1].trim()}`,
    `starting_stacks = [${d.stacks.map((s) => s.trim()).join(", ")}]`,
    `actions = [${actions}]`,
    `players = [${seats.map((s) => `'${s}'`).join(", ")}]`,
    "",
  ].join("\n")
}

function saidBefore(seen: Seen): Step[] {
  const order = seen.names.map((_, i) => (FIRST_AFTER_BLINDS + i) % seen.names.length)
  const steps: Step[] = []
  for (const seat of order) {
    const verb = seen.in_hand[seat] ? SAID[seen.said[seat].trim().toLowerCase().split(" ")[0]] : "f"
    if (seat === seen.hero || !verb) break
    steps.push({ seat, verb, amount: "" })
  }
  return steps
}

export function drafted(seen: Seen, blinds: [string, string]): Draft {
  const [flop, turn, river] = [seen.board.slice(0, 3), seen.board.slice(3, 4), seen.board.slice(4, 5)]
  return {
    hero: seen.hero,
    cards: seen.cards.join(" "),
    board: { flop: flop.join(" "), turn: turn.join(" "), river: river.join(" ") },
    steps: { ...NO_STEPS, preflop: seen.board.length ? [] : saidBefore(seen) },
    names: seen.names,
    stacks: seen.stacks.map(String),
    blinds,
  }
}

export const ready = (d: Draft) => packed(d.cards).length === 4

export const betting = (street: Street) => `${street[0].toUpperCase()}${street.slice(1)} betting`
