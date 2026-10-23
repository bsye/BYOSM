import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { OpenLibrary } from "./OpenLibrary"
import { OpenSearch } from "./OpenSearch"
import { TrainingData } from "./TrainingData"

export function Datasets() {
  return (
    <div className="flex flex-col gap-4 md:gap-6">
      <h1 className="text-2xl font-semibold">Datasets</h1>
      <Tabs defaultValue="training">
        <TabsList>
          <TabsTrigger value="training">Training data</TabsTrigger>
          <TabsTrigger value="open">Open data</TabsTrigger>
          <TabsTrigger value="library">Kept</TabsTrigger>
        </TabsList>
        <TabsContent value="training">
          <TrainingData />
        </TabsContent>
        <TabsContent value="open">
          <OpenSearch />
        </TabsContent>
        <TabsContent value="library">
          <OpenLibrary />
        </TabsContent>
      </Tabs>
    </div>
  )
}
