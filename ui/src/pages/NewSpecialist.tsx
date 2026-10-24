import { Hammer } from "lucide-react"
import { useState } from "react"
import { runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { openJob } from "@/job-sheet"

const EXAMPLE = "classify support emails into billing, shipping, technical or other"

export function NewSpecialist({ children, initial = "" }: { children: React.ReactNode; initial?: string }) {
  const [prompt, setPrompt] = useState(initial)
  const [error, setError] = useState("")
  const start = () =>
    runJob(["auto", prompt])
      .then((job) => openJob(job.id))
      .catch((e) => setError(String(e)))
  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New specialist</DialogTitle>
          <DialogDescription>
            Describe the task in one sentence. The autopilot infers the labels, finds a benchmark and rivals, and trains until it wins or the budget ends.
          </DialogDescription>
        </DialogHeader>
        <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={EXAMPLE} />
        {error && <p className="text-destructive text-sm">{error}</p>}
        <DialogFooter>
          <Button disabled={!prompt.trim()} onClick={start}>
            <Hammer />
            Build it
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
