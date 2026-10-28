import type { CSSProperties, ReactNode } from "react"

export function WordSwitch({
  word,
  off,
  tone,
  title,
  onClick,
  children,
}: {
  word: string
  off: boolean
  tone?: string
  title?: string
  onClick: () => void
  children?: ReactNode
}) {
  const style: CSSProperties = { backgroundColor: off || !tone ? "transparent" : tone }
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`hover:ring-ring cursor-pointer rounded px-1 text-sm transition-colors hover:ring-1 ${off ? "text-muted-foreground line-through" : ""}`}
      style={style}
    >
      {word}
      {children}
    </button>
  )
}
