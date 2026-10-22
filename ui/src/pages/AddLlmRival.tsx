import { Bot } from "lucide-react"
import { useState } from "react"
import { api } from "@/api"
import { LlmPicker, useLlmChoice } from "@/components/llm-picker"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

export function AddLlmRival({ name, onAdded }: { name: string; onAdded: () => void }) {
  const llm = useLlmChoice()
  const [open, setOpen] = useState(false)
  const [error, setError] = useState("")
  const add = async () => {
    try {
      await api(`/specialists/${name}/llm-rivals`, { method: "POST", body: JSON.stringify({ config: llm.config() }) })
      setOpen(false)
      onAdded()
    } catch (e) {
      setError(String(e))
    }
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <Bot />
          Add LLM
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Add an LLM rival</DialogTitle>
        </DialogHeader>
        <LlmPicker llm={llm} label="LLM to compare against" />
        {error && <p className="text-destructive text-sm">{error}</p>}
        <DialogFooter>
          <Button disabled={!llm.choice} onClick={add}>
            Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
