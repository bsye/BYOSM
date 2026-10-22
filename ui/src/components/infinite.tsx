import { Loader2 } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

export const PAGE_SIZE = 100
const SCROLL_VIEWPORT = "[data-slot=scroll-area-viewport]"

export type Page<T, X> = { total: number; rows: T[]; extra: X }

export function useInfinite<T, X>(key: string, fetchPage: (offset: number, limit: number) => Promise<Page<T, X>>, merge: (a: X, b: X) => X, refresh = 0) {
  const [rows, setRows] = useState<T[]>([])
  const [extra, setExtra] = useState<X | null>(null)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const loaded = useRef(0)
  const reload = useCallback(
    (limit: number) => {
      setLoading(true)
      fetchPage(0, limit)
        .then((p) => {
          setRows(p.rows)
          setExtra(p.extra)
          setTotal(p.total)
          loaded.current = p.rows.length
        })
        .finally(() => setLoading(false))
    },
    [fetchPage],
  )
  useEffect(() => {
    loaded.current = 0
    reload(PAGE_SIZE)
  }, [key, reload])
  useEffect(() => {
    if (refresh) reload(Math.max(PAGE_SIZE, loaded.current))
  }, [refresh, reload])
  const more = () => {
    if (loading || loaded.current >= total) return
    setLoading(true)
    fetchPage(loaded.current, PAGE_SIZE)
      .then((p) => {
        setRows((r) => [...r, ...p.rows])
        setExtra((e) => (e ? merge(e, p.extra) : p.extra))
        setTotal(p.total)
        loaded.current += p.rows.length
      })
      .finally(() => setLoading(false))
  }
  return { rows, extra, total, loading, more, done: rows.length >= total }
}

export function LoadMore({ onVisible, loading, done, shown, total }: { onVisible: () => void; loading: boolean; done: boolean; shown: number; total: number }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || done) return
    const root = el.closest(SCROLL_VIEWPORT)
    const watch = new IntersectionObserver((seen) => seen[0]?.isIntersecting && onVisible(), { root })
    watch.observe(el)
    return () => watch.disconnect()
  }, [onVisible, done])
  return (
    <div ref={ref} className="text-muted-foreground flex items-center justify-center gap-2 py-3 text-xs tabular-nums">
      {loading && <Loader2 className="size-3.5 animate-spin" />}
      {shown.toLocaleString()} of {total.toLocaleString()}
    </div>
  )
}
