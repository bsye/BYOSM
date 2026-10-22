import { Mic, Square } from "lucide-react"
import { Room, RoomEvent } from "livekit-client"
import { useEffect, useRef, useState } from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import { post } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartLegend, ChartLegendContent, type ChartConfig } from "@/components/ui/chart"
import { ms } from "@/format"

const TOPIC = "vad"
const WINDOW_POINTS = 300
const DECODER = new TextDecoder()

type Point = { t: number; ours: number | null; ours_speaking: boolean; silero: number; silero_speaking: boolean; ours_ms: number; silero_ms: number }
type Session = { url: string; token: string; room: string; job: string }

const config = {
  ours: { label: "this model", color: "var(--chart-1)" },
  silero: { label: "Silero", color: "var(--chart-3)" },
} satisfies ChartConfig

function useLiveRoom() {
  const room = useRef<Room | null>(null)
  const [points, setPoints] = useState<Point[]>([])
  const [state, setState] = useState<"idle" | "connecting" | "live">("idle")
  const [error, setError] = useState("")
  useEffect(() => () => void room.current?.disconnect(), [])
  const start = async (name: string) => {
    setError("")
    setState("connecting")
    setPoints([])
    try {
      const s = await post<Session>(`/specialists/${name}/live`, {})
      const r = new Room()
      r.on(RoomEvent.DataReceived, (payload, _who, _kind, topic) => {
        if (topic !== TOPIC) return
        const p = JSON.parse(DECODER.decode(payload)) as Point
        setPoints((old) => [...old, p].slice(-WINDOW_POINTS))
      })
      r.on(RoomEvent.Disconnected, () => setState("idle"))
      await r.connect(s.url, s.token)
      await r.localParticipant.setMicrophoneEnabled(true)
      room.current = r
      setState("live")
    } catch (e) {
      setError(String(e))
      setState("idle")
    }
  }
  const stop = () => {
    void room.current?.disconnect()
    room.current = null
    setState("idle")
  }
  return { points, state, error, start, stop }
}

function Speaking({ label, on }: { label: string; on: boolean | undefined }) {
  return <Badge variant={on ? "default" : "outline"}>{`${label}: ${on ? "speech" : "silence"}`}</Badge>
}

export function LiveVad({ name }: { name: string }) {
  const live = useLiveRoom()
  const last = live.points.at(-1)
  const t0 = live.points[0]?.t ?? 0
  const data = live.points.map((p) => ({ ...p, s: Number((p.t - t0).toFixed(1)) }))
  const mean = (k: "ours_ms" | "silero_ms") => (live.points.length ? live.points.reduce((a, p) => a + p[k], 0) / live.points.length : null)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Live microphone
          <InfoTip>Your microphone goes through LiveKit to this model and to Silero, LiveKit's own VAD, side by side.</InfoTip>
        </CardTitle>
        <CardAction>
          {live.state === "idle" ? (
            <Button size="sm" onClick={() => live.start(name)}>
              <Mic />
              Start
            </Button>
          ) : (
            <Button size="sm" variant="outline" onClick={live.stop} disabled={live.state === "connecting"}>
              <Square />
              {live.state === "connecting" ? "Connecting" : "Stop"}
            </Button>
          )}
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        {live.error && <p className="text-destructive text-sm">{live.error}</p>}
        <div className="flex flex-wrap items-center gap-2">
          <Speaking label="This model" on={last?.ours_speaking} />
          <Speaking label="Silero" on={last?.silero_speaking} />
          {last && (
            <span className="text-muted-foreground ml-auto text-xs tabular-nums">
              {ms(mean("ours_ms"))} vs {ms(mean("silero_ms"))} per 32 ms chunk
            </span>
          )}
        </div>
        <ChartContainer config={config} className="h-56 w-full">
          <LineChart data={data} margin={{ left: 0, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="s" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} unit=" s" />
            <YAxis domain={[0, 1]} tickLine={false} axisLine={false} width={32} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line dataKey="ours" stroke="var(--color-ours)" dot={false} isAnimationActive={false} strokeWidth={2} />
            <Line dataKey="silero" stroke="var(--color-silero)" dot={false} isAnimationActive={false} strokeWidth={2} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
