import { useQuery } from "@tanstack/react-query";
import { getDriftStatus } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/utils";
import { AlertTriangle, CheckCircle } from "lucide-react";

export function DriftStatus() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["drift-status"],
    queryFn: getDriftStatus,
  });

  const driftedCount = data?.filter((d) => d.is_drifted).length ?? 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900">Drift Durumu</h3>
          {data && (
            <span
              className={cn(
                "inline-flex items-center gap-1 text-xs font-medium",
                driftedCount > 0 ? "text-amber-600" : "text-emerald-600",
              )}
            >
              {driftedCount > 0 ? (
                <>
                  <AlertTriangle size={12} />
                  {driftedCount} drift
                </>
              ) : (
                <>
                  <CheckCircle size={12} />
                  Normal
                </>
              )}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-6 w-full" />
            ))}
          </div>
        )}

        {error && <p className="text-sm text-rose-500">Veri yuklenemedi</p>}

        {data && data.length > 0 && (
          <div className="space-y-2">
            {data.map((d) => (
              <div
                key={d.feature}
                className={cn(
                  "flex items-center justify-between px-3 py-2 rounded-sm text-sm",
                  d.is_drifted ? "bg-amber-50" : "bg-slate-50",
                )}
              >
                <span className="text-slate-700 text-xs">{d.feature}</span>
                <span
                  className={cn(
                    "font-mono text-xs font-medium",
                    d.is_drifted ? "text-amber-600" : "text-slate-500",
                  )}
                >
                  {d.drift_score.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}

        {data && data.length === 0 && (
          <p className="text-sm text-slate-400">Drift verisi yok</p>
        )}
      </CardContent>
    </Card>
  );
}
