import { useQuery } from "@tanstack/react-query";
import { getSystemHealth } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/utils";
import { Activity } from "lucide-react";

export function SystemHealth() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["system-health"],
    queryFn: getSystemHealth,
    refetchInterval: 30_000,
  });

  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <Activity size={16} className="text-slate-400" />
          Sistem Sagligi
        </h3>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-6 w-full" />
            ))}
          </div>
        )}

        {error && <p className="text-sm text-rose-500">Veri yuklenemedi</p>}

        {data && (
          <div className="space-y-3">
            <HealthRow
              label="Veritabani"
              value={data.database === "connected" ? "Bagli" : "Bagli Degil"}
              ok={data.database === "connected"}
            />
            <HealthRow
              label="Modeller"
              value={data.models_loaded ? "Yuklu" : "Yuklenmedi"}
              ok={data.models_loaded}
            />
            <HealthRow
              label="Tahmin Ufku"
              value={`${data.model_info.forecast_horizon} saat`}
              ok={true}
            />
            <div>
              <p className="text-xs text-slate-500 mb-1.5">Aktif Modeller</p>
              <div className="flex flex-wrap gap-1.5">
                {data.model_info.active_models.map((m) => (
                  <span
                    key={m}
                    className="inline-block px-2 py-0.5 text-xs font-medium bg-emerald-50 text-emerald-700 rounded-sm"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1.5">Model Agirliklari</p>
              <div className="space-y-1">
                {Object.entries(data.model_info.weights).map(([model, weight]) => (
                  <div key={model} className="flex items-center gap-2">
                    <span className="text-xs text-slate-600 w-16">{model}</span>
                    <div className="flex-1 bg-slate-100 rounded-full h-1.5">
                      <div
                        className="h-1.5 rounded-full bg-emerald-500"
                        style={{ width: `${(weight as number) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono text-slate-500 w-10 text-right">
                      {((weight as number) * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function HealthRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500">{label}</span>
      <span
        className={cn(
          "text-xs font-medium font-mono",
          ok ? "text-emerald-600" : "text-rose-500",
        )}
      >
        {value}
      </span>
    </div>
  );
}
