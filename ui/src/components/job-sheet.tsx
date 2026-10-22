import { Maximize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { closeJob, useOpenJob } from "@/job-sheet"
import { JobDetail } from "@/pages/JobLog"
import { href } from "@/route"

export function JobSheet() {
  const id = useOpenJob()
  return (
    <Sheet open={id != null} onOpenChange={(open) => !open && closeJob()}>
      <SheetContent side="right" className="w-full overflow-y-auto data-[side=right]:sm:max-w-2xl">
        <SheetHeader className="flex-row items-center justify-between pr-12">
          <SheetTitle className="text-muted-foreground text-sm font-normal">Job {id}</SheetTitle>
          <SheetDescription className="sr-only">The job you started, live.</SheetDescription>
          {id && (
            <Button size="sm" variant="outline" asChild>
              <a href={href("jobs", id)} onClick={closeJob}>
                <Maximize2 />
                Full page
              </a>
            </Button>
          )}
        </SheetHeader>
        <div className="@container/main px-4 pb-6">{id && <JobDetail id={id} />}</div>
      </SheetContent>
    </Sheet>
  )
}
