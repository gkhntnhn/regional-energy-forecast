import { cn } from "@/lib/utils";
import type { JobStatus } from "@/api/types";

const statusConfig: Record<JobStatus, { dot: string; text: string; label: string }> = {
  pending: { dot: "bg-amber-500", text: "text-amber-700", label: "Kuyrukta" },
  processing: { dot: "bg-emerald-500 animate-pulse", text: "text-emerald-700", label: "Isleniyor" },
  completed: { dot: "bg-emerald-500", text: "text-emerald-700", label: "Tamamlandi" },
  failed: { dot: "bg-rose-500", text: "text-rose-700", label: "Basarisiz" },
  cancelled: { dot: "bg-slate-400", text: "text-slate-600", label: "Iptal" },
  running: { dot: "bg-emerald-500 animate-pulse", text: "text-emerald-700", label: "Calisiyor" },
  queued: { dot: "bg-amber-500", text: "text-amber-700", label: "Kuyrukta" },
};

interface BadgeProps {
  status: JobStatus;
  className?: string;
}

export function StatusBadge({ status, className }: BadgeProps) {
  const config = statusConfig[status];
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs font-medium", config.text, className)}>
      <span className={cn("w-2 h-2 rounded-full", config.dot)} />
      {config.label}
    </span>
  );
}
