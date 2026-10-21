import { Activity, Boxes, KeyRound, Network } from "lucide-react"
import { useState } from "react"
import { saveToken } from "@/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Toaster } from "@/components/ui/sonner"
import { LiveProvider, useLive, useJobToasts } from "@/events"
import { Jobs, Running } from "@/pages/Jobs"
import { Keys } from "@/pages/Keys"
import { Palette } from "@/pages/Palette"
import { Specialist } from "@/pages/Specialist"
import { Specialists } from "@/pages/Specialists"
import { Systems } from "@/pages/Systems"
import { href, useRoute } from "@/route"

const NAV = [
  { key: "specialists", label: "Specialists", icon: Boxes },
  { key: "systems", label: "Systems", icon: Network },
  { key: "jobs", label: "Jobs", icon: Activity },
  { key: "keys", label: "Keys", icon: KeyRound },
]

function Page({ route }: { route: string[] }) {
  const [section = "specialists", id] = route
  if (section === "specialists") return id ? <Specialist name={id} /> : <Specialists />
  if (section === "systems") return <Systems />
  if (section === "jobs") return <Jobs id={id} />
  if (section === "keys") return <Keys />
  return <Specialists />
}

function TokenPrompt() {
  const [value, setValue] = useState("")
  return (
    <Alert variant="destructive">
      <AlertTitle>This page has an old token</AlertTitle>
      <AlertDescription className="space-y-3">
        <p>Open the link `byosm app` printed, or paste it (or just the token) here.</p>
        <div className="flex gap-2">
          <Input value={value} onChange={(e) => setValue(e.target.value)} placeholder="http://127.0.0.1:8765/#token=..." />
          <Button disabled={!value.trim()} onClick={() => saveToken(value)}>
            Use it
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  )
}

function Shell() {
  const route = useRoute()
  const live = useLive()
  useJobToasts()
  const running = Object.values(live.jobs).filter((j) => j.status === "running" || j.status === "queued").length
  const section = route[0] ?? "specialists"
  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader className="px-4 py-3 text-lg font-semibold">byosm</SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV.map((n) => (
                  <SidebarMenuItem key={n.key}>
                    <SidebarMenuButton asChild isActive={section === n.key}>
                      <a href={href(n.key)}>
                        <n.icon />
                        <span>{n.label}</span>
                      </a>
                    </SidebarMenuButton>
                    {n.key === "jobs" && running > 0 && <SidebarMenuBadge>{running}</SidebarMenuBadge>}
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
      <SidebarInset>
        <header className="flex h-14 items-center gap-3 border-b px-4">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-5" />
          <span className="text-sm">{route.join(" / ") || "specialists"}</span>
          <kbd className="text-muted-foreground ml-auto hidden rounded border px-1.5 font-mono text-xs md:inline">⌘K</kbd>
          <Badge variant={live.connected ? "secondary" : "destructive"}>
            {live.connected ? "live" : "offline"}
          </Badge>
        </header>
        <main className="space-y-4 p-4 md:p-6">
          {live.unauthorized && <TokenPrompt />}
          <Running />
          <Page route={route} />
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}

export default function App() {
  return (
    <LiveProvider>
      <Shell />
      <Palette />
      <Toaster />
    </LiveProvider>
  )
}
