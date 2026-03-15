import { useQuery } from "@tanstack/react-query";
import { UploadForm } from "@/components/predict/UploadForm";
import { JobQueue } from "@/components/predict/JobQueue";
import { ResultTable } from "@/components/predict/ResultTable";
import { ResultChart } from "@/components/predict/ResultChart";
import { useJobStore } from "@/stores/jobs";
import { getJobPredictions } from "@/api/predict";
import { Skeleton } from "@/components/ui/Skeleton";
import { useMemo } from "react";
import { useActiveJobs } from "@/hooks/useActiveJobs";

export function DashboardPage() {
  useActiveJobs();
  const lastResult = useJobStore((s) => s.lastResult);
  const jobId = lastResult?.job_id;

  const { data: predictions, isLoading } = useQuery({
    queryKey: ["job-predictions", jobId],
    queryFn: () => getJobPredictions(jobId!),
    enabled: !!jobId && lastResult?.status === "completed",
  });

  const stats = useMemo(() => {
    if (!predictions || predictions.length === 0) return undefined;
    const values = predictions.map((p) => p.consumption_mwh);
    const sum = values.reduce((a, b) => a + b, 0);
    return {
      count: values.length,
      mean: sum / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [predictions]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Dashboard</h2>
        <p className="text-sm text-slate-500">Tahmin olustur ve sonuclari goruntule</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <UploadForm />
        </div>
        <JobQueue />
      </div>

      {/* Result section */}
      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-72 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      )}

      {predictions && predictions.length > 0 && (
        <div className="space-y-6">
          <ResultChart predictions={predictions} />
          <ResultTable
            predictions={predictions}
            stats={stats}
          />
        </div>
      )}
    </div>
  );
}
