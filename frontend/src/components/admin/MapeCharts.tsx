import { useQuery } from "@tanstack/react-query";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getDailyMape, getWeeklyMape, getHourlyMape } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { useState } from "react";

type Period = "daily" | "weekly" | "hourly";

const periodLabel: Record<Period, string> = {
  daily: "Gunluk",
  weekly: "Haftalik",
  hourly: "Saatlik",
};

export function MapeCharts() {
  const [period, setPeriod] = useState<Period>("daily");

  const daily = useQuery({ queryKey: ["mape-daily"], queryFn: getDailyMape, enabled: period === "daily" });
  const weekly = useQuery({ queryKey: ["mape-weekly"], queryFn: getWeeklyMape, enabled: period === "weekly" });
  const hourly = useQuery({ queryKey: ["mape-hourly"], queryFn: getHourlyMape, enabled: period === "hourly" });

  const isLoading = period === "daily" ? daily.isLoading : period === "weekly" ? weekly.isLoading : hourly.isLoading;
  const error = period === "daily" ? daily.error : period === "weekly" ? weekly.error : hourly.error;

  return (
    <Card accent>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900">MAPE Performansi</h3>
          <div className="flex bg-slate-100 rounded-sm p-0.5">
            {(["daily", "weekly", "hourly"] as Period[]).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-3 py-1 text-xs font-medium rounded-sm transition-colors cursor-pointer ${
                  period === p
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {periodLabel[p]}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-64 w-full" />}
        {error && <p className="text-sm text-rose-500">Veri yuklenemedi</p>}

        {period === "daily" && daily.data && (
          <MapeAreaChart data={daily.data.map((d) => ({ label: d.date, mape: d.mape }))} />
        )}
        {period === "weekly" && weekly.data && (
          <MapeAreaChart data={weekly.data.map((d) => ({ label: d.date, mape: d.mape }))} />
        )}
        {period === "hourly" && hourly.data && (
          <MapeBarChart data={hourly.data.map((d) => ({ label: `${d.hour}:00`, mape: d.mape }))} />
        )}
      </CardContent>
    </Card>
  );
}

function MapeAreaChart({ data }: { data: { label: string; mape: number }[] }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 8, left: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="mapeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "#94a3b8" }}
            axisLine={{ stroke: "#e2e8f0" }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
            width={45}
            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
          />
          <Tooltip content={<MapeTooltip />} />
          <Area
            type="monotone"
            dataKey="mape"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#mapeGradient)"
            dot={false}
            activeDot={{ r: 4, fill: "#10b981", stroke: "#fff", strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function MapeBarChart({ data }: { data: { label: string; mape: number }[] }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: "#94a3b8" }}
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
          <Tooltip content={<MapeTooltip />} />
          <Bar dataKey="mape" fill="#10b981" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function MapeTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { label: string; mape: number } }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-slate-900 text-white rounded-sm px-3 py-2 text-xs shadow-lg">
      <p className="font-mono">{d.label}</p>
      <p className="font-mono font-medium mt-0.5">MAPE: {d.mape.toFixed(2)}%</p>
    </div>
  );
}
