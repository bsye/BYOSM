import { ExternalLink } from "lucide-react"
import { InfoTip } from "@/components/info-tip"

export type ModelRef = { id: string; name: string; detail: string; url: string | null }

export function ModelName({ model, shown }: { model: ModelRef | undefined; shown: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      {model?.url ? (
        <a className="font-medium hover:underline" href={model.url} target="_blank" rel="noreferrer">
          {shown}
          <ExternalLink className="text-muted-foreground ml-1 inline size-3" />
        </a>
      ) : (
        <span className="font-medium">{shown}</span>
      )}
      {model?.detail && <InfoTip>{model.detail}</InfoTip>}
    </span>
  )
}
