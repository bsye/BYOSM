import { Save } from "lucide-react"
import { useState } from "react"
import { put, type Detail } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"

export function SpecEditor({ d, onSaved }: { d: Detail; onSaved: () => void }) {
  const [text, setText] = useState(JSON.stringify(d.spec, null, 2))
  const [note, setNote] = useState("")
  const save = async () => {
    try {
      await put(`/specialists/${d.name}/spec`, JSON.parse(text))
      setNote("saved")
      onSaved()
    } catch (e) {
      setNote(String(e))
    }
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>spec.yaml</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea className="h-[32rem] font-mono text-xs" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="flex items-center gap-3">
          <Button onClick={save}>
            <Save />
            Save changes
          </Button>
          <span className="text-muted-foreground text-sm">{note}</span>
        </div>
      </CardContent>
    </Card>
  )
}
