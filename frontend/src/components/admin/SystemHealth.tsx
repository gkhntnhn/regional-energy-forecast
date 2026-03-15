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

  function formatUptime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return h > 0 ? `${h}s ${m}dk` : `${m}dk`;
  }

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
              label="Durum"
              value={data.status}
              ok={data.status === "healthy"}
            />
            <HealthRow
              label="Uptime"
              value={formatUptime(data.uptime_seconds)}
              ok={true}
            />
            <HealthRow
              label="Kuyruk"
              value={`${data.queue_size} is`}
              ok={data.queue_size < 5}
            />
            <HealthRow
              label="Veritabani"
              value={data.db_connected ? "Bagli" : "Bagli Degil"}
              ok={data.db_connected}
            />
            <div>
              <p className="text-xs text-slate-500 mb-1.5">Yuklu Modeller</p>
              <div className="flex flex-wrap gap-1.5">
                {data.models_loaded.map((m) => (
                  <span
                    key={m}
                    className="inline-block px-2 py-0.5 text-xs font-medium bg-emerald-50 text-emerald-700 rounded-sm"
                  >
                    {m}
                  </span>
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
