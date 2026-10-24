import { Plus } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"

const SAFE = "-_"

const slug = (text: string) =>
  [...text.trim().toLowerCase()]
    .map((c) => (c === " " ? "-" : c))
    .filter((c) => (c >= "a" && c <= "z") || (c >= "0" && c <= "9") || SAFE.includes(c))
    .join("")

export function NewTestSet({ taken, onCreate }: { taken: string[]; onCreate: (name: string) => void }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState("")
  const name = slug(text)
  const clash = taken.includes(name)
  const create = () => {
    onCreate(name)
    setText("")
    setOpen(false)
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <Plus />
          New set
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>New test set</DialogTitle>
        </DialogHeader>
        <Input
          autoFocus
          placeholder="e.g. noisy-calls"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && name && !clash && create()}
        />
        {clash && <p className="text-destructive text-sm">{name} already exists</p>}
        <DialogFooter>
          <Button disabled={!name || clash} onClick={create}>
            Create {name}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
