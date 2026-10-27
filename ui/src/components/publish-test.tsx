import { FlaskConical, Loader2 } from "lucide-react"
import { useState } from "react"
import { post } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"

type Answer = { label: string; probs?: Record<string, number> }
type Trial = { n: number; right: number; ms: number; examples: { input: string; label: string; answer: string }[]; answer: Answer | null }

const pct = (v: number) => `${Math.round(v * 100)}%`

export function PublishTest({ name }: { name: string }) {
  const [text, setText] = useState("")
  const [trial, setTrial] = useState<Trial | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const run = () => {
    setBusy(true)
    setError("")
    post<Trial>(`/specialists/${name}/publish/test`, { text: text.trim() || null })
      .then(setTrial)
      .catch((e) => setError(String(e)))
      .finally(() => setBusy(false))
  }
  const sure = trial?.answer?.probs?.[trial.answer.label]
  return (
    <div className="grid gap-3">
      <Textarea rows={3} value={text} placeholder="Optional: an input of your own, in the card's input format" onChange={(e) => setText(e.target.value)} />
      <div>
        <Button variant="outline" onClick={run} disabled={busy}>
          {busy ? <Loader2 className="animate-spin" /> : <FlaskConical />}
          Test the bundle
        </Button>
      </div>
      {error && <p className="text-destructive text-sm">{error}</p>}
      {trial && (
        <div className="grid gap-2 text-sm">
          <p>
            {trial.right} of {trial.n} test inputs right ({pct(trial.right / Math.max(1, trial.n))}), {trial.ms} ms each on CPU
          </p>
          {trial.answer && (
            <p>
              Your input: <Badge>{trial.answer.label}</Badge> {sure != null && <span className="text-muted-foreground">{pct(sure)} sure</span>}
            </p>
          )}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Input</TableHead>
                <TableHead>Label</TableHead>
                <TableHead>Answer</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trial.examples.map((e, i) => (
                <TableRow key={i}>
                  <TableCell className="max-w-72 truncate" title={e.input}>
                    {e.input}
                  </TableCell>
                  <TableCell>{e.label}</TableCell>
                  <TableCell>
                    <Badge variant={e.answer === e.label ? "secondary" : "destructive"}>{e.answer}</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
