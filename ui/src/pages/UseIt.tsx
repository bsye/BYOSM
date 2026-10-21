import { Check, Container, Copy } from "lucide-react"
import { useState } from "react"
import { runJob, type Detail } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { go } from "@/route"

const MODEL_PORT = 8800
const COPIED_MS = 1500

function Snippet({ title, code, action }: { title: string; code: string; action?: React.ReactNode }) {
  const [copied, setCopied] = useState(false)
  const copy = () =>
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), COPIED_MS)
    })
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardAction className="flex gap-2">
          {action}
          <Button variant="ghost" size="icon" aria-label="Copy" onClick={copy}>
            {copied ? <Check /> : <Copy />}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        <pre className="bg-muted overflow-x-auto rounded-md p-3 font-mono text-xs leading-5">{code}</pre>
      </CardContent>
    </Card>
  )
}

export function UseIt({ d }: { d: Detail }) {
  if (!d.exported) return <p className="text-muted-foreground text-sm">Export {d.name} first.</p>
  const dir = `specialists/${d.name}/export`
  const image = `byosm-${d.name.toLowerCase()}`
  const example = d.modality === "text" ? "your text here" : "path/to/input"
  const build = () => runJob(["containerize", d.name]).then((j) => go("jobs", j.id))
  return (
    <div className="grid auto-rows-fr gap-4 @3xl/main:grid-cols-2">
      <Snippet
        title="Python"
        code={`pip install numpy onnxruntime tokenizers\n\nimport sys; sys.path.insert(0, "${dir}")\nfrom byosm_rt import load\nmodel = load("${dir}")\nmodel.predict("${example}")`}
      />
      <Snippet title="Command line" code={`python ${dir}/byosm_rt.py predict ${dir} "${example}"`} />
      <Snippet
        title="HTTP"
        code={`byosm serve ${d.name} --port ${MODEL_PORT}\n\ncurl -d '{"input": "${example}"}' \\\n  http://127.0.0.1:${MODEL_PORT}/v1/${d.name}/predict`}
      />
      <Snippet
        title="Container"
        code={`byosm containerize ${d.name}\ndocker run --rm --read-only -p ${MODEL_PORT}:${MODEL_PORT} ${image}`}
        action={
          <Button variant="outline" size="sm" onClick={build}>
            <Container />
            Build image
          </Button>
        }
      />
    </div>
  )
}
