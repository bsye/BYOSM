import { Activity, Boxes, CircleDollarSign, Container, Database, Cpu, Github, House, Settings, Smile, TrendingUp } from "lucide-react"
import type { ComponentProps } from "react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { href } from "@/route"

const HOME = [{ key: "home", label: "Home", icon: House }]
const GROUPS = [
  {
    label: "Build",
    items: [
      { key: "specialists", label: "Specialists", icon: Boxes },
      { key: "forecasts", label: "Forecasts", icon: TrendingUp },
      { key: "datasets", label: "Datasets", icon: Database },
    ],
  },
  {
    label: "Run",
    items: [
      { key: "jobs", label: "Jobs", icon: Activity },
      { key: "deployments", label: "Deployments", icon: Container },
    ],
  },
  {
    label: "Spend",
    items: [
      { key: "trainers", label: "Trainers", icon: Cpu },
      { key: "costs", label: "Costs", icon: CircleDollarSign },
    ],
  },
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
  const item = (n: (typeof HOME)[number]) => (
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
            <SidebarMenu>{HOME.map(item)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        {GROUPS.map((g) => (
          <SidebarGroup key={g.label}>
            <SidebarGroupLabel>{g.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>{g.items.map(item)}</SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
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
