import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DataFinder } from "./DataFinder"
import { Forecasts } from "./Forecasts"
import { OpenLibrary } from "./OpenLibrary"
import { Sources } from "./Sources"
import { TrainingData } from "./TrainingData"

export function Datasets() {
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Datasets</h1>
      <Tabs defaultValue="find">
        <TabsList>
          <TabsTrigger value="find">Find</TabsTrigger>
          <TabsTrigger value="library">Library</TabsTrigger>
          <TabsTrigger value="sources">Sources</TabsTrigger>
          <TabsTrigger value="training">Training data</TabsTrigger>
          <TabsTrigger value="forecasts">Forecasts</TabsTrigger>
        </TabsList>
        <TabsContent value="find">
          <DataFinder />
        </TabsContent>
        <TabsContent value="library">
          <OpenLibrary />
        </TabsContent>
        <TabsContent value="sources">
          <Sources />
        </TabsContent>
        <TabsContent value="training">
          <TrainingData />
        </TabsContent>
        <TabsContent value="forecasts">
          <Forecasts />
        </TabsContent>
      </Tabs>
    </div>
  )
}
