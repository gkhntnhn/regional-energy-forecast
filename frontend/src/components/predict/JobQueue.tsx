import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { JobStatusBar } from "./JobStatus";
import { useJobStore } from "@/stores/jobs";
import { useJobPolling } from "@/hooks/useJobPolling";
import { Loader2 } from "lucide-react";

function PollingJob({ jobId }: { jobId: string }) {
  useJobPolling(jobId);
  return null;
}

export function JobQueue() {
  const activeJobs = useJobStore((s) => s.activeJobs);

  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          Aktif Isler
          {activeJobs.length > 0 && (
            <span className="flex items-center gap-1 text-xs font-normal text-emerald-600">
              <Loader2 size={12} className="animate-spin" />
              {activeJobs.length}
            </span>
          )}
        </h3>
      </CardHeader>
      <CardContent>
        {activeJobs.length === 0 ? (
          <p className="text-sm text-slate-400">Aktif is yok</p>
        ) : (
          <div className="space-y-3">
            {activeJobs.map((job) => (
              <div key={job.job_id}>
                <PollingJob jobId={job.job_id} />
                <JobStatusBar job={job} />
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
