import { toast } from "sonner"
import { api, type Job, type Summary } from "@/api"
import { rivalName, score, signed, took } from "@/format"
import { openJob } from "@/job-sheet"
import { describe } from "@/jobs"

async function standing(target: string | null): Promise<string> {
  if (!target) return ""
  const s = (await api<Summary[]>("/specialists").catch(() => [])).find((x) => x.name === target)
  if (!s || s.score == null) return ""
  const lead = s.rival?.score != null ? `, ${signed(s.score - s.rival.score)} vs ${rivalName(s.rival.name)}` : ""
  return `${s.title || s.name}: ${score(s.score)} test ${s.metric}${lead}. `
}

export async function announce(j: Job) {
  const d = describe(j)
  const title = [d.title, d.target].filter(Boolean).join(" ")
  const action = { label: "Open", onClick: () => openJob(j.id) }
  const time = took(j.started, j.ended)
  if (j.status === "done") toast.success(title, { description: `${await standing(d.target)}Took ${time}.`, action })
  else
    toast.error(title, {
      description: `${j.status === "cancelled" ? "Cancelled" : j.status === "interrupted" ? "Interrupted" : "Failed"} after ${time}.`,
      action,
    })
}
