import { Badge } from "@/components/ui/badge"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { href } from "@/route"
import { ThemeToggle } from "./theme"

export function SiteHeader({ route, connected }: { route: string[]; connected: boolean }) {
  const parts = route.length ? route : ["create"]
  return (
    <header className="flex h-(--header-height) shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mx-2 data-[orientation=vertical]:h-4" />
        <Breadcrumb>
          <BreadcrumbList>
            {parts.map((p, i) => (
              <span key={i} className="contents">
                {i > 0 && <BreadcrumbSeparator />}
                <BreadcrumbItem>
                  {i === parts.length - 1 ? (
                    <BreadcrumbPage>{p}</BreadcrumbPage>
                  ) : (
                    <BreadcrumbLink href={href(...parts.slice(0, i + 1))}>{p}</BreadcrumbLink>
                  )}
                </BreadcrumbItem>
              </span>
            ))}
          </BreadcrumbList>
        </Breadcrumb>
        <div className="ml-auto flex items-center gap-2">
          <kbd className="bg-muted text-muted-foreground hidden h-5 items-center rounded border px-1.5 font-mono text-[10px] font-medium md:inline-flex">
            ⌘K
          </kbd>
          <Badge variant={connected ? "outline" : "destructive"}>{connected ? "Live" : "Offline"}</Badge>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
