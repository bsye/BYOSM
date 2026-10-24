import { AudioLines, ChartLine, FileText, Highlighter, type LucideIcon, Tags, Users } from "lucide-react"
import type { Summary } from "@/api"

export type Kind = { label: string; plural: string; icon: LucideIcon }

const KINDS: Record<string, Kind> = {
  team: { label: "Team", plural: "Teams", icon: Users },
  audio: { label: "Audio", plural: "Audio", icon: AudioLines },
  document: { label: "Document", plural: "Documents", icon: FileText },
  tagging: { label: "Tagging", plural: "Text tagging", icon: Highlighter },
  text: { label: "Text", plural: "Text classification", icon: Tags },
  forecast: { label: "Forecast", plural: "Forecasting", icon: ChartLine },
}
const KIND_ORDER = ["text", "tagging", "document", "audio", "forecast", "team"]

const DAY_MS = 24 * 60 * 60 * 1000
const WEEK_DAYS = 7
const UNDATED = "Undated"

export function kindOf(s: Summary): Kind {
  if (s.kind === "team") return KINDS.team
  if (s.kind === "forecast") return KINDS.forecast
  if (s.modality === "audio") return KINDS.audio
  if (s.modality === "document" || s.modality === "image") return KINDS.document
  return s.task_type === "frame" ? KINDS.tagging : KINDS.text
}

export function leadOf(s: Summary): number | null {
  if (s.score == null || s.rival?.score == null) return null
  return s.higher_is_better === false ? s.rival.score - s.score : s.score - s.rival.score
}

export const titleOf = (s: Summary) => s.title || s.name.split("-").join(" ")

const createdAt = (s: Summary) => (s.created ? new Date(s.created).getTime() : 0)

export const newestFirst = (items: Summary[]) => [...items].sort((a, b) => createdAt(b) - createdAt(a))

function dayGroup(s: Summary, now: Date): string {
  if (!s.created) return UNDATED
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const days = Math.floor((today - new Date(s.created).setHours(0, 0, 0, 0)) / DAY_MS)
  if (days <= 0) return "Today"
  if (days === 1) return "Yesterday"
  if (days < WEEK_DAYS) return "This week"
  return new Date(s.created).toLocaleDateString(undefined, { month: "long", year: "numeric" })
}

export type Group = { key: string; label: string; items: Summary[] }

function grouped(items: Summary[], keyOf: (s: Summary) => string): Map<string, Summary[]> {
  const out = new Map<string, Summary[]>()
  for (const s of newestFirst(items)) out.set(keyOf(s), [...(out.get(keyOf(s)) ?? []), s])
  return out
}

export function byKind(items: Summary[]): Group[] {
  const groups = grouped(items, (s) => KIND_ORDER.find((k) => KINDS[k] === kindOf(s)) ?? "text")
  return KIND_ORDER.filter((k) => groups.has(k)).map((k) => ({ key: k, label: KINDS[k].plural, items: groups.get(k) ?? [] }))
}

export function byDate(items: Summary[], now = new Date()): Group[] {
  return [...grouped(items, (s) => dayGroup(s, now))].map(([label, list]) => ({ key: label, label, items: list }))
}

export function createdLabel(s: Summary, now = new Date()): string {
  if (!s.created) return ""
  const when = new Date(s.created)
  const group = dayGroup(s, now)
  const time = when.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  if (group === "Today" || group === "Yesterday") return `${group}, ${time}`
  return when.toLocaleDateString(undefined, { day: "numeric", month: "short" })
}
