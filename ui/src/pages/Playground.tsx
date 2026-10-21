import { Play } from "lucide-react"
import { useState } from "react"
import { post, type Detail } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Textarea } from "@/components/ui/textarea"

type Prediction = { label?: string; probs?: Record<string, number>; entities?: { label: string; text: string }[] }

const SHOWN = 5

export function Playground({ d }: { d: Detail }) {
  const [text, setText] = useState("")
  const [out, setOut] = useState<Prediction | null>(null)
  const [error, setError] = useState("")
  if (!d.exported) return <p className="text-muted-foreground text-sm">Export {d.name} to try it here.</p>
  if (d.modality !== "text") return <p className="text-muted-foreground text-sm">The playground takes text for now.</p>
  const run = () =>
    post<Prediction>(`/specialists/${d.name}/predict`, { input: text })
      .then((p) => {
        setOut(p)
        setError("")
      })
      .catch((e) => setError(String(e)))
  const top = Object.entries(out?.probs ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, SHOWN)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Try it</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="type an input" />
        <Button disabled={!text.trim()} onClick={run}>
          <Play />
          Predict
        </Button>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {out?.label && <p className="text-lg font-semibold">{out.label}</p>}
        {top.map(([label, p]) => (
          <div key={label} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span>{label}</span>
              <span className="tabular-nums">{p.toFixed(3)}</span>
            </div>
            <Progress value={p * 100} />
          </div>
        ))}
        {out?.entities?.map((e, i) => (
          <p key={i} className="text-sm">
            <span className="font-medium">{e.label}</span>: {e.text}
          </p>
        ))}
      </CardContent>
    </Card>
  )
}
