import { useQuery } from "@tanstack/react-query";
import { UploadForm } from "@/components/predict/UploadForm";
import { JobQueue } from "@/components/predict/JobQueue";
import { ResultTable } from "@/components/predict/ResultTable";
import { ResultChart } from "@/components/predict/ResultChart";
import { useJobStore } from "@/stores/jobs";
import { getJobResult } from "@/api/predict";
import { Skeleton } from "@/components/ui/Skeleton";

export function DashboardPage() {
  const lastResult = useJobStore((s) => s.lastResult);
  const jobId = lastResult?.job_id;

  const { data: result, isLoading } = useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => getJobResult(jobId!),
    enabled: !!jobId && lastResult?.status === "completed",
  });

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

      {result?.predictions && result.predictions.length > 0 && (
        <div className="space-y-6">
          <ResultChart predictions={result.predictions} />
          <ResultTable
            predictions={result.predictions}
            downloadUrl={result.download_url}
            stats={result.statistics}
          />
        </div>
      )}
    </div>
  );
}
