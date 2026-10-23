import { ExternalLink, Upload } from "lucide-react"
import { useEffect, useState } from "react"
import { api, runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { openJob } from "@/job-sheet"

const REPO_PREFIX = "byosm-"
const PRIVATE = "private"
const PUBLIC = "public"

function useHubUser(open: boolean): string | null {
  const [user, setUser] = useState<string | null>(null)
  useEffect(() => {
    if (!open) return
    api<{ user: string | null }>("/hub/user")
      .then((r) => setUser(r.user))
      .catch(() => setUser(null))
  }, [open])
  return user
}

type Props = {
  name: string
  exported: boolean
  published?: string
  label?: string
  size?: "sm" | "default"
  command?: (repo: string, pub: boolean) => string[]
}

const publishModel = (name: string) => (repo: string, pub: boolean) => ["publish", name, "--repo", repo, ...(pub ? ["--public"] : [])]

export function PublishDialog({ name, exported, published, label = "Publish", size = "default", command }: Props) {
  const [open, setOpen] = useState(false)
  const user = useHubUser(open)
  const [repo, setRepo] = useState("")
  const [visibility, setVisibility] = useState(PRIVATE)
  const [error, setError] = useState("")
  const suggested = published ?? (user ? `${user}/${REPO_PREFIX}${name}` : "")
  const target = repo || suggested
  const publish = () =>
    runJob((command ?? publishModel(name))(target, visibility === PUBLIC))
      .then((j) => openJob(j.id))
      .catch((e) => setError(String(e)))
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size={size} disabled={!exported} title={exported ? undefined : "Export first"}>
          <Upload />
          {label}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Publish to Hugging Face</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="repo">Repository</Label>
            <Input id="repo" value={target} placeholder="you/byosm-model" onChange={(e) => setRepo(e.target.value)} />
            {!user && open && <p className="text-muted-foreground text-xs">Not logged in: add HF_TOKEN in Settings, Keys.</p>}
          </div>
          <ToggleGroup type="single" variant="outline" value={visibility} onValueChange={(v) => v && setVisibility(v)}>
            <ToggleGroupItem value={PRIVATE}>Private</ToggleGroupItem>
            <ToggleGroupItem value={PUBLIC}>Public</ToggleGroupItem>
          </ToggleGroup>
          {error && <p className="text-destructive text-sm">{error}</p>}
        </div>
        <DialogFooter>
          {published && (
            <Button variant="outline" asChild>
              <a href={`https://huggingface.co/${published}`} target="_blank" rel="noreferrer">
                <ExternalLink />
                Open
              </a>
            </Button>
          )}
          <Button disabled={!target.includes("/")} onClick={publish}>
            <Upload />
            Publish
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
