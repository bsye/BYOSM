import { Play } from "lucide-react"
import { runJob } from "@/api"
import { InfoTip } from "@/components/info-tip"
import { StatCard, STAT_GRID } from "@/components/stat-card"
import { Button } from "@/components/ui/button"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useLive } from "@/events"
import { ms, rivalName, score, signed } from "@/format"
import { openJob } from "@/job-sheet"
import { trainTime, type ForecasterView } from "./forecaster-api"
import { Board } from "./ForecasterBoard"
import { Datasets } from "./ForecasterDatasets"
import { Versions } from "./ForecasterVersions"
import { Health } from "./Health"
import { Lab } from "./Lab"
import { StepsCard } from "./NextSteps"
import { RunningHere } from "./RunningHere"
import { SeriesTry } from "./SeriesTry"
import { leadOf, rivalScope, titleOf } from "./specialist-groups"

function Stats({ f }: { f: ForecasterView }) {
  const promoted = f.versions?.find((v) => v.name === f.promoted)
  if (!promoted) return null
  const recipe = f.overall?.[promoted.recipe]
  return (
    <div className={STAT_GRID}>
      <StatCard
        label={`Test ${f.metric}`}
        value={score(promoted.test)}
        detail={promoted.ci ? `95% CI ${score(promoted.ci[0])} to ${score(promoted.ci[1])}` : promoted.name}
      />
      <StatCard
        label="Lead over best rival"
        value={signed(leadOf(f))}
        detail={f.rival ? `vs ${rivalName(f.rival.name)}${rivalScope(f)}, lower is better` : "no rival scored yet"}
      />
      <StatCard
        label="Training"
        value={trainTime(recipe?.train_s)}
        detail={`${promoted.recipe}, one model for ${Object.keys(f.configs ?? {}).length} datasets`}
      />
      <StatCard label="Per forecast" value={ms(recipe?.ms_per_forecast)} detail="CPU" />
    </div>
  )
}

export function Forecaster({ f, tab }: { f: ForecasterView; tab?: string }) {
  const running = Object.values(useLive().jobs).some((j) => j.args[0] === "series" && j.args[2] === f.name && (j.status === "running" || j.status === "queued"))
  const train = () => void runJob(["series", "run", f.name, "--until-rival"]).then((j) => openJob(j.id))
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            {titleOf(f)}
            <InfoTip>
              One model per recipe, trained on all of these GIFT-Eval datasets together. MASE and CRPS are relative to seasonal naive (1.0, the dashed line):
              lower is better. Each run picks the recipe on the calib windows and reports the test windows. Published scores come from the GIFT-Eval leaderboard
              and show only where our seasonal naive matches theirs.
            </InfoTip>
          </h1>
          <p className="text-muted-foreground text-sm">{f.description}</p>
        </div>
        <Button onClick={train}>
          <Play />
          Keep training
        </Button>
      </div>
      <RunningHere name={f.name} />
      <div className="grid gap-4 *:data-[slot=card]:h-full @3xl/main:grid-cols-2">
        <Health h={f.health} />
        <StepsCard name={f.name} steps={f.next ?? []} proposals={[]} ask={["lab", "write", f.name]} batch={false} />
      </div>
      <Stats f={f} />
      {f.overall && Object.keys(f.overall).length ? (
        <Tabs defaultValue={tab ?? "board"}>
          <TabsList>
            <TabsTrigger value="board">Board</TabsTrigger>
            <TabsTrigger value="try">Try it</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
            <TabsTrigger value="lab">Lab</TabsTrigger>
            <TabsTrigger value="datasets">Datasets</TabsTrigger>
          </TabsList>
          <TabsContent value="board">
            <Board f={f} running={running} />
          </TabsContent>
          <TabsContent value="try">
            <SeriesTry f={f} />
          </TabsContent>
          <TabsContent value="versions">
            <Versions f={f} />
          </TabsContent>
          <TabsContent value="lab">
            <Lab name={f.name} kind="forecaster" />
          </TabsContent>
          <TabsContent value="datasets">
            <Datasets f={f} />
          </TabsContent>
        </Tabs>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>Not run yet</EmptyTitle>
            <EmptyDescription>Keep training builds the first version and scores it against the rivals.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      )}
    </div>
  )
}
