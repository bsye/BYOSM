const CHAR_PX = 7
const AXIS_PAD_PX = 12

export const ROW_PX = 28

export const axisWidth = (labels: string[]) => Math.max(...labels.map((l) => l.length)) * CHAR_PX + AXIS_PAD_PX

export const INTERVAL_BAR = { width: 4, strokeWidth: 1, stroke: "var(--foreground)" }
