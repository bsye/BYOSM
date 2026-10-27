import { ExternalLink, Upload } from "lucide-react"
import { useEffect, useState } from "react"
import { api, post, runJob } from "@/api"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { openJob } from "@/job-sheet"
import { PublishCard } from "./publish-card"
import { PublishTest } from "./publish-test"

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
const modelExtras = (card: string, data: boolean) => [...(card ? ["--card", card] : []), ...(data ? [] : ["--no-data"])]

export function PublishDialog({ name, exported, published, label = "Publish", size = "default", command }: Props) {
  const [open, setOpen] = useState(false)
  const user = useHubUser(open)
  const [repo, setRepo] = useState("")
  const [visibility, setVisibility] = useState(PRIVATE)
  const [error, setError] = useState("")
  const [card, setCard] = useState("")
  const [hasData, setHasData] = useState(false)
  const [withData, setWithData] = useState(true)
  const isModel = command == null
  const suggested = published ?? (user ? `${user}/${REPO_PREFIX}${name}` : "")
  const target = repo || suggested
  const saved = () => (card ? post<{ path: string }>(`/specialists/${name}/publish/card`, { repo: target, card }).then((r) => r.path) : Promise.resolve(""))
  const publish = () =>
    saved()
      .then((path) => runJob([...(command ?? publishModel(name))(target, visibility === PUBLIC), ...(isModel ? modelExtras(path, withData || !hasData) : [])]))
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
      <DialogContent className={isModel ? "sm:max-w-3xl" : undefined}>
        <DialogHeader>
          <DialogTitle>Publish to Hugging Face</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="repo">Repository</Label>
            <Input id="repo" value={target} placeholder="you/byosm-model" onChange={(e) => setRepo(e.target.value)} />
            {!user && open && <p className="text-muted-foreground text-xs">Not logged in: add HF_TOKEN in Settings, Keys.</p>}
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <ToggleGroup type="single" variant="outline" value={visibility} onValueChange={(v) => v && setVisibility(v)}>
              <ToggleGroupItem value={PRIVATE}>Private</ToggleGroupItem>
              <ToggleGroupItem value={PUBLIC}>Public</ToggleGroupItem>
            </ToggleGroup>
            {isModel && hasData && (
              <Label className="font-normal">
                <Checkbox checked={withData} onCheckedChange={(v) => setWithData(v === true)} />
                Include the dataset
              </Label>
            )}
          </div>
          {isModel && (
            <Tabs defaultValue="card">
              <TabsList>
                <TabsTrigger value="card">Card</TabsTrigger>
                <TabsTrigger value="test">Test</TabsTrigger>
              </TabsList>
              <TabsContent value="card">
                <PublishCard name={name} repo={target} card={card} setCard={setCard} onPreview={(p) => setHasData(p.data)} />
              </TabsContent>
              <TabsContent value="test">
                <PublishTest name={name} />
              </TabsContent>
            </Tabs>
          )}
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
