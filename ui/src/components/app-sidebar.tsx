import { Activity, Boxes, KeyRound, Network, Settings, Sparkles } from "lucide-react"
import type { ComponentProps } from "react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { href } from "@/route"

const MAIN = [
  { key: "specialists", label: "Specialists", icon: Boxes },
  { key: "systems", label: "Systems", icon: Network },
  { key: "jobs", label: "Jobs", icon: Activity },
]
const SECONDARY = [
  { key: "keys", label: "Keys", icon: KeyRound },
  { key: "settings", label: "Settings", icon: Settings },
]

type Props = ComponentProps<typeof Sidebar> & { section: string; running: number }

export function AppSidebar({ section, running, ...props }: Props) {
  const item = (n: (typeof MAIN)[number]) => (
    <SidebarMenuItem key={n.key}>
      <SidebarMenuButton asChild isActive={section === n.key} tooltip={n.label}>
        <a href={href(n.key)}>
          <n.icon />
          <span>{n.label}</span>
        </a>
      </SidebarMenuButton>
      {n.key === "jobs" && running > 0 && <SidebarMenuBadge>{running}</SidebarMenuBadge>}
    </SidebarMenuItem>
  )
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild className="data-[slot=sidebar-menu-button]:p-1.5!">
              <a href={href("specialists")}>
                <Sparkles className="size-5!" />
                <span className="text-base font-semibold">byosm</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>{MAIN.map(item)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup className="mt-auto">
          <SidebarGroupContent>
            <SidebarMenu>{SECONDARY.map(item)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <p className="text-muted-foreground px-2 text-xs">Small specialists, local, no LLM at runtime.</p>
      </SidebarFooter>
    </Sidebar>
  )
}
