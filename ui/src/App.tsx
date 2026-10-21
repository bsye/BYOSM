import { KeyRound } from "lucide-react"
import { useState, type CSSProperties } from "react"
import { saveToken } from "@/api"
import { AppSidebar } from "@/components/app-sidebar"
import { ErrorBoundary } from "@/components/error-boundary"
import { SiteHeader } from "@/components/site-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { LiveProvider, useLive, useJobToasts } from "@/events"
import { Create } from "@/pages/Create"
import { Jobs, Running } from "@/pages/Jobs"
import { Keys } from "@/pages/Keys"
import { Palette } from "@/pages/Palette"
import { Settings } from "@/pages/Settings"
import { Specialist } from "@/pages/Specialist"
import { Specialists } from "@/pages/Specialists"
import { Systems } from "@/pages/Systems"
import { useRoute } from "@/route"

function Page({ route }: { route: string[] }) {
  const [section = "create", id] = route
  if (section === "create") return <Create />
  if (section === "specialists") return id ? <Specialist name={id} /> : <Specialists />
  if (section === "systems") return <Systems />
  if (section === "jobs") return <Jobs id={id} />
  if (section === "keys") return <Keys />
  if (section === "settings") return <Settings />
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
            <KeyRound />
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
  return (
    <SidebarProvider
      style={{ "--sidebar-width": "calc(var(--spacing) * 64)", "--header-height": "calc(var(--spacing) * 12)" } as CSSProperties}
    >
      <AppSidebar variant="inset" section={route[0] ?? "create"} running={running} />
      <SidebarInset>
        <SiteHeader route={route} connected={live.connected} />
        <div className="flex flex-1 flex-col">
          <div className="@container/main flex flex-1 flex-col gap-2">
            <div className="flex flex-col gap-4 px-4 py-4 md:gap-6 md:py-6 lg:px-6">
              {live.unauthorized && <TokenPrompt />}
              <Running />
              <ErrorBoundary resetKey={route.join("/")}>
                <Page route={route} />
              </ErrorBoundary>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

export default function App() {
  return (
    <LiveProvider>
      <TooltipProvider>
        <Shell />
      </TooltipProvider>
      <Palette />
      <Toaster />
    </LiveProvider>
  )
}
