import { cn } from "@/lib/utils";
import { StatusBadge } from "@/components/ui/Badge";
import type { Job } from "@/api/types";

interface JobStatusProps {
  job: Job;
}

export function JobStatusBar({ job }: JobStatusProps) {
  const progress = typeof job.progress === "number" ? job.progress : 0;
  const isActive = job.status === "processing" || job.status === "pending";

  return (
    <div className="flex items-center gap-3">
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-mono text-slate-500">
            {job.job_id.slice(0, 8)}
          </span>
          <StatusBadge status={job.status} />
        </div>
        <div className="w-full bg-slate-100 rounded-full h-1.5">
          <div
            className={cn(
              "h-1.5 rounded-full transition-all duration-500",
              job.status === "failed" ? "bg-rose-500" : "bg-emerald-500",
              isActive && progress < 100 && "animate-pulse",
            )}
            style={{ width: `${Math.max(progress, isActive ? 5 : 0)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
