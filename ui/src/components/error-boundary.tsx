import { RotateCcw, TriangleAlert } from "lucide-react"
import { Component, type ReactNode } from "react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"

type Props = { children: ReactNode; resetKey: string }
type State = { error: Error | null; key: string }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, key: this.props.resetKey }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  static getDerivedStateFromProps(props: Props, state: State) {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <Alert variant="destructive">
        <TriangleAlert />
        <AlertTitle>This page failed to render</AlertTitle>
        <AlertDescription className="space-y-3">
          <p className="font-mono text-xs">{this.state.error.message}</p>
          <Button variant="outline" size="sm" onClick={() => this.setState({ error: null })}>
            <RotateCcw />
            Try again
          </Button>
        </AlertDescription>
      </Alert>
    )
  }
}
