import { Activity, Boxes, CircleDollarSign, Container, Database, Cpu, Github, House, Settings, Smile, TrendingUp } from "lucide-react"
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
  { key: "home", label: "Home", icon: House },
  { key: "specialists", label: "Specialists", icon: Boxes },
  { key: "jobs", label: "Jobs", icon: Activity },
  { key: "models", label: "Models", icon: Cpu },
  { key: "datasets", label: "Datasets", icon: Database },
  { key: "forecasts", label: "Forecasts", icon: TrendingUp },
  { key: "deployments", label: "Deployments", icon: Container },
  { key: "costs", label: "Costs", icon: CircleDollarSign },
]
const SECONDARY = [
  { key: "settings", label: "Settings", icon: Settings },
]

const LINKS = [
  { label: "GitHub", handle: "bsye", href: "https://github.com/bsye", icon: Github },
  { label: "Hugging Face", handle: "bsye", href: "https://huggingface.co/bsye", icon: Smile },
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
              <a href={href("home")}>
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
        <SidebarMenu>
          {LINKS.map((l) => (
            <SidebarMenuItem key={l.href}>
              <SidebarMenuButton asChild size="sm" tooltip={l.label}>
                <a href={l.href} target="_blank" rel="noreferrer">
                  <l.icon />
                  <span>{l.label}</span>
                  <span className="text-muted-foreground/70 ml-auto text-xs">{l.handle}</span>
                </a>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
