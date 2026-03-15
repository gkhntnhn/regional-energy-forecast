import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getModelMape } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { formatMape } from "@/lib/utils";

export function ModelComparison() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["model-mape"],
    queryFn: getModelMape,
  });

  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-900">Model Karsilastirma</h3>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-64 w-full" />}
        {error && <p className="text-sm text-rose-500">Veri yuklenemedi</p>}

        {data && data.length > 0 && (
          <>
            <div className="h-48 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data} margin={{ top: 4, right: 8, left: 8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis
                    dataKey="model"
                    tick={{ fontSize: 12, fill: "#334155" }}
                    axisLine={{ stroke: "#e2e8f0" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "#94a3b8" }}
                    axisLine={false}
                    tickLine={false}
                    width={45}
                    tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.[0]) return null;
                      const d = payload[0].payload as { model: string; mape: number; mae: number; rmse: number };
                      return (
                        <div className="bg-slate-900 text-white rounded-sm px-3 py-2 text-xs shadow-lg">
                          <p className="font-medium">{d.model}</p>
                          <p className="font-mono mt-1">MAPE: {d.mape.toFixed(2)}%</p>
                          <p className="font-mono">MAE: {d.mae.toFixed(1)}</p>
                          <p className="font-mono">RMSE: {d.rmse.toFixed(1)}</p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="mape" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Table */}
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-2 text-xs font-medium text-slate-500">Model</th>
                  <th className="text-right py-2 text-xs font-medium text-slate-500">MAPE</th>
                  <th className="text-right py-2 text-xs font-medium text-slate-500">MAE</th>
                  <th className="text-right py-2 text-xs font-medium text-slate-500">RMSE</th>
                </tr>
              </thead>
              <tbody>
                {data.map((m) => (
                  <tr key={m.model} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                    <td className="py-2 font-medium text-slate-900">{m.model}</td>
                    <td className="py-2 text-right font-mono text-xs">{formatMape(m.mape)}</td>
                    <td className="py-2 text-right font-mono text-xs text-slate-600">{m.mae.toFixed(1)}</td>
                    <td className="py-2 text-right font-mono text-xs text-slate-600">{m.rmse.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {data && data.length === 0 && (
          <p className="text-sm text-slate-400">Henuz model verisi yok</p>
        )}
      </CardContent>
    </Card>
  );
}
