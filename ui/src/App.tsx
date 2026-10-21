import { Activity, Boxes, KeyRound, Network } from "lucide-react"
import { Badge } from "@/components/ui/badge"
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
          <span className="text-sm capitalize">{route.join(" / ") || "specialists"}</span>
          <Badge variant={live.connected ? "secondary" : "destructive"} className="ml-auto">
            {live.connected ? "live" : "offline"}
          </Badge>
        </header>
        <main className="space-y-4 p-4 md:p-6">
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
      <Toaster />
    </LiveProvider>
  )
}
