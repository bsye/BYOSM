import { useEffect, useState } from "react";
import { api } from "@/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useLive } from "@/events";
import { seconds, took } from "@/format";
import { describe } from "@/jobs";
import { go } from "@/route";

const REFRESH_MS = 10000;

type Holder = {
  slot: number;
  share?: number;
  job?: string | null;
  since?: number;
};
type Waiter = { job: string; waited: number };
type Gpu = { holders: Holder[]; waiting: Waiter[] };

function useGpu() {
  const [gpu, setGpu] = useState<Gpu | null>(null);
  useEffect(() => {
    const load = () => void api<Gpu>("/gpu").then(setGpu, () => undefined);
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, []);
  return gpu;
}

export function GpuQueue() {
  const gpu = useGpu();
  const jobs = useLive().jobs;
  if (!gpu || gpu.holders.length + gpu.waiting.length === 0) return null;
  const name = (id?: string | null) => {
    const job = id ? jobs[id] : undefined;
    if (!job) return "Another process";
    const d = describe(job);
    return [d.title, d.target ?? d.subject].filter(Boolean).join(" ");
  };
  const rows = [
    ...gpu.holders.map((h) => ({
      key: `slot-${h.slot}`,
      job: h.job,
      state: "Training",
      share: h.share,
      time: h.since ? took(h.since, null) : "-",
    })),
    ...gpu.waiting.map((w) => ({
      key: w.job,
      job: w.job,
      state: "Waiting",
      share: undefined,
      time: seconds(w.waited),
    })),
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>GPU</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-32">State</TableHead>
              <TableHead>Job</TableHead>
              <TableHead className="text-right">Share</TableHead>
              <TableHead className="text-right">For</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow
                key={r.key}
                className={r.job ? "cursor-pointer" : undefined}
                onClick={() => r.job && go("jobs", r.job)}
              >
                <TableCell>
                  <Badge
                    variant={r.state === "Training" ? "default" : "outline"}
                  >
                    {r.state}
                  </Badge>
                </TableCell>
                <TableCell className="font-medium">{name(r.job)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {r.share == null ? "-" : `${Math.round(r.share * 100)}%`}
                </TableCell>
                <TableCell className="text-foreground/70 text-right tabular-nums">
                  {r.time}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
