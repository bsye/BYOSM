import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DataFinder } from "./DataFinder"
import { OpenLibrary } from "./OpenLibrary"
import { Sources } from "./Sources"
import { TrainingData } from "./TrainingData"
import { DatasetGaps } from "./DatasetGaps"

export function Datasets() {
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Datasets</h1>
      <Tabs defaultValue="find">
        <TabsList>
          <TabsTrigger value="find">Find</TabsTrigger>
          <TabsTrigger value="gaps">Gaps</TabsTrigger>
          <TabsTrigger value="library">Library</TabsTrigger>
          <TabsTrigger value="sources">Sources</TabsTrigger>
          <TabsTrigger value="training">Training data</TabsTrigger>
        </TabsList>
        <TabsContent value="find">
          <DataFinder />
        </TabsContent>
        <TabsContent value="gaps">
          <DatasetGaps />
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
      </Tabs>
    </div>
  )
}
