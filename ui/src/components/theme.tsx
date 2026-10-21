import { Monitor, Moon, Sun } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

type Theme = "light" | "dark" | "system"

const KEY = "byosm-theme"
const DARK_QUERY = "(prefers-color-scheme: dark)"

function stored(): Theme {
  try {
    return (localStorage.getItem(KEY) as Theme) || "system"
  } catch {
    return "system"
  }
}

function apply(theme: Theme) {
  const dark = theme === "dark" || (theme === "system" && window.matchMedia(DARK_QUERY).matches)
  document.documentElement.classList.toggle("dark", dark)
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(stored)
  useEffect(() => {
    apply(theme)
    const media = window.matchMedia(DARK_QUERY)
    const on = () => apply(theme)
    media.addEventListener("change", on)
    return () => media.removeEventListener("change", on)
  }, [theme])
  const set = (t: Theme) => {
    try {
      localStorage.setItem(KEY, t)
    } catch {
      return setTheme(t)
    }
    setTheme(t)
  }
  return [theme, set]
}

const ICONS = { light: Sun, dark: Moon, system: Monitor }

export function ThemeToggle() {
  const [theme, setTheme] = useTheme()
  const Icon = ICONS[theme]
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Theme">
          <Icon />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {(["light", "dark", "system"] as Theme[]).map((t) => (
          <DropdownMenuItem key={t} onClick={() => setTheme(t)}>
            {t === "system" ? "System" : t === "light" ? "Light" : "Dark"}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
