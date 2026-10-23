import { Loader2, Repeat, Upload } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"
import { runJob, upload, type Detail } from "@/api"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { InfoTip } from "@/components/info-tip"
import { openJob } from "@/job-sheet"
import { score } from "@/format"

const EPOCHS = ["1", "2", "3", "5"]
const DEFAULT_EPOCHS = "3"

export function ContinueDialog({ d }: { d: Detail }) {
  const trained = d.runs.filter((r) => d.board[r.name]?.[d.primary_suite] && !r.ablation_arm)
  const [open, setOpen] = useState(false)
  const [start, setStart] = useState(d.promoted ?? "")
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS)
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const from = start || d.promoted || trained[0]?.name || ""
  const go = async () => {
    setBusy(true)
    try {
      const added = file ? await upload<{ path: string }>(`/specialists/${d.name}/uploads`, file) : null
      const job = await runJob(["continue", d.name, "--from", from, "--epochs", epochs, ...(added ? ["--data", added.path] : []), "--run"])
      setOpen(false)
      openJob(job.id)
    } catch (e) {
      toast.error(String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" disabled={!trained.length} title={trained.length ? undefined : "Train it first"}>
          <Repeat />
          Keep training
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Keep training
            <InfoTip>Starts from the trained weights at half the learning rate, adds your rows and replays the old ones, so it learns the new data without forgetting. It is promoted only if it beats its parent on calib.</InfoTip>
          </DialogTitle>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid gap-1.5">
            <Label>Continue from</Label>
            <Select value={from} onValueChange={setStart}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {trained.map((r) => (
                  <SelectItem key={r.name} value={r.name}>
                    {r.name}
                    {r.name === d.promoted ? " (promoted)" : ""} · calib {score(d.board[r.name][d.primary_suite]?.calib_objective)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="continue-data" className="flex items-center gap-2">
              New rows
              <InfoTip>Optional. A table with the data template's columns (text, label, split); it is checked for format, leaks and likely mislabels before training.</InfoTip>
            </Label>
            <Input id="continue-data" type="file" accept=".csv,.tsv,.xlsx,.jsonl" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <div className="grid gap-1.5">
            <Label>Epochs</Label>
            <Select value={epochs} onValueChange={setEpochs}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EPOCHS.map((e) => (
                  <SelectItem key={e} value={e}>
                    {e}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button disabled={!from || busy} onClick={() => void go()}>
            {busy ? <Loader2 className="animate-spin" /> : file ? <Upload /> : <Repeat />}
            {file ? "Add rows and train" : "Train more"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
