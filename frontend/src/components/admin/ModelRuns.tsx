import { useQuery } from "@tanstack/react-query";
import { getTrainingRuns } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { formatDateTime, formatMape } from "@/lib/utils";
import { cn } from "@/lib/utils";

export function ModelRuns() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["training-runs"],
    queryFn: getTrainingRuns,
  });

  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-900">Training Run'lari</h3>
      </CardHeader>
      <CardContent className="p-0">
        {isLoading && (
          <div className="p-5 space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        )}

        {error && <p className="p-5 text-sm text-rose-500">Veri yuklenemedi</p>}

        {data && data.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/50">
                  <th className="text-left px-5 py-2 text-xs font-medium text-slate-500">Model</th>
                  <th className="text-left px-5 py-2 text-xs font-medium text-slate-500">Durum</th>
                  <th className="text-left px-5 py-2 text-xs font-medium text-slate-500">Tarih</th>
                  <th className="text-right px-5 py-2 text-xs font-medium text-slate-500">Val MAPE</th>
                  <th className="text-right px-5 py-2 text-xs font-medium text-slate-500">Test MAPE</th>
                  <th className="text-right px-5 py-2 text-xs font-medium text-slate-500">Sure</th>
                </tr>
              </thead>
              <tbody>
                {data.map((run) => (
                  <tr
                    key={run.id}
                    className="border-b border-slate-100 hover:bg-slate-50 transition-colors duration-150"
                  >
                    <td className="px-5 py-2">
                      <span className="inline-block px-2 py-0.5 text-xs font-medium bg-slate-100 text-slate-700 rounded-sm">
                        {run.model_type}
                      </span>
                    </td>
                    <td className="px-5 py-2">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1 text-xs font-medium",
                          run.status === "completed" ? "text-emerald-600" : "text-amber-600",
                        )}
                      >
                        <span
                          className={cn(
                            "w-1.5 h-1.5 rounded-full",
                            run.status === "completed" ? "bg-emerald-500" : "bg-amber-500",
                          )}
                        />
                        {run.status}
                      </span>
                    </td>
                    <td className="px-5 py-2 text-xs text-slate-600">
                      {formatDateTime(run.started_at)}
                    </td>
                    <td className="px-5 py-2 text-right font-mono text-xs font-medium text-slate-900">
                      {run.val_mape != null ? formatMape(run.val_mape) : "-"}
                    </td>
                    <td className="px-5 py-2 text-right font-mono text-xs text-slate-600">
                      {run.test_mape != null ? formatMape(run.test_mape) : "-"}
                    </td>
                    <td className="px-5 py-2 text-right font-mono text-xs text-slate-500">
                      {run.duration_seconds != null ? `${run.duration_seconds}s` : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data && data.length === 0 && (
          <p className="p-5 text-sm text-slate-400">Henuz training run'i yok</p>
        )}
      </CardContent>
    </Card>
  );
}
